import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax

# ---------- Device ----------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ---------- Encoders ----------
class EdgeEncoder(nn.Module):
    """
    输入 edge_feat(u,v) = [
        deg(u), mean_deg(u), type(u),
        deg(v), mean_deg(v), type(v),
        tx_lat, prop_lat
    ] → ℝ^D
    归一化：对度/均值/时延统一做 log1p（type 为 0/1，log1p 也无害）。
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_edge_full: torch.Tensor) -> torch.Tensor:
        return self.net(x_edge_full)


class SubchunkEncoder(nn.Module):
    """
    输入 subchunk_feat(s) = [ pos_in_node(s), deg(src), mean_deg(src), type(src) ] → ℝ^D
    仅对后 3 项做 log1p；pos 已在 [0,1]。
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_sc_full: torch.Tensor) -> torch.Tensor:
        return self.net(x_sc_full)


# ---------- Graph Transformer ----------
class GraphTransformer(nn.Module):
    """
    Graph Transformer for edge representations using PyTorch's TransformerEncoder.
    Processes edge features [E, D] -> [E, D] using self-attention.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, edge_repr: torch.Tensor, attn_mask=None):
        # edge_repr: [E, D]
        # Transformer expects [batch, seq_len, d_model] if batch_first=True
        x = edge_repr.unsqueeze(0)  # [1, E, D]
        # NOTE: PyTorch TransformerEncoder expects mask of shape [E, E] (not per-head).
        # Pass the provided float mask directly; 0.0 for allowed, -inf for disallowed.
        x_out = self.encoder(x, mask=attn_mask)     # [1, E, D]
        # Remove batch dimension
        x_out = x_out.squeeze(0)    # [E, D]
        return x_out




# --------- Sparsemax 激活函数 ----------
class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1. 排序
        input_sorted, _ = torch.sort(input, descending=True, dim=self.dim)
        input_cumsum = input_sorted.cumsum(dim=self.dim)

        # 2. 计算支持集
        rhos = torch.arange(1, input.size(self.dim)+1, device=input.device, dtype=input.dtype)
        shape = [1] * input.dim()
        shape[self.dim] = -1
        rhos = rhos.view(shape)

        support = (1 + rhos * input_sorted) > input_cumsum
        k = support.sum(dim=self.dim, keepdim=True)

        # 3. 计算 tau
        tau = (input_cumsum.gather(self.dim, k-1) - 1) / k

        # 4. 计算输出
        output = torch.clamp(input - tau, min=0)
        return output



# ---------- Scoring Head（输出稠密矩阵，输出 logits） ----------
class LogitScoringHead(nn.Module):
    """
    Produces a full [S, E] matrix of **logits** for all subchunks and edges.
    These logits are unnormalized scores; do NOT use sigmoid/softmax here.
    Downstream (env or rollout code) can apply mask + softmax/top-k to obtain weights.
    """
    def __init__(self, hidden_dim: int, proj_dim: int = 64, scale: float = 5.0):
        super().__init__()
        self.proj_sub = nn.Linear(hidden_dim, proj_dim, bias=True)
        self.proj_edge = nn.Linear(hidden_dim, proj_dim, bias=True)
        # learnable temperature-like scale for stability/flexibility
        self.log_scale = nn.Parameter(torch.tensor(float(scale)).log())

        nn.init.xavier_uniform_(self.proj_sub.weight)
        nn.init.xavier_uniform_(self.proj_edge.weight)
        nn.init.zeros_(self.proj_sub.bias)
        nn.init.zeros_(self.proj_edge.bias)

    def forward(self, edge_repr: torch.Tensor, sub_repr: torch.Tensor) -> torch.Tensor:
        # sub_repr: [S, D], edge_repr: [E, D]
        sub_proj = F.normalize(self.proj_sub(sub_repr), dim=-1)    # [S, P]
        edge_proj = F.normalize(self.proj_edge(edge_repr), dim=-1) # [E, P]
        # Dot-product similarity scaled by learnable factor → logits [S, E]
        scale = self.log_scale.exp().clamp(min=0.1, max=50.0)
        logits = torch.matmul(sub_proj, edge_proj.t()) * scale  # [S, E]
        return logits

# ---------- Value Head (Critic) ----------
class ValueHead(nn.Module):
    """Graph-level value head that outputs a scalar V(s).
    It pools token representations (subchunks + edges) and maps to a single value.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, sub_repr: torch.Tensor, edge_repr: torch.Tensor) -> torch.Tensor:
        # sub_repr: [S, D], edge_repr: [E, D]
        if sub_repr.numel() == 0 and edge_repr.numel() == 0:
            raise ValueError("ValueHead received empty representations")
        if sub_repr.numel() == 0:
            pooled = edge_repr.mean(dim=0, keepdim=True)         # [1, D]
        elif edge_repr.numel() == 0:
            pooled = sub_repr.mean(dim=0, keepdim=True)          # [1, D]
        else:
            pooled = torch.cat([sub_repr, edge_repr], dim=0).mean(dim=0, keepdim=True)  # [1, D]
        v = self.mlp(pooled).squeeze(-1)  # [1]
        return v


# ---------- Agent ----------
class AllGatherCritics(nn.Module):
    """
    __init__(nodes, edges, subchunks_node):
      - 三份字典直接进来；内部预处理为张量
    forward():
      - 编码 → 图变换 → 池化 → **输出标量 V(s)**（评论家，只看状态，不在这里处理动作/softmax）。
    """
    def __init__(self, nodes: dict, edges: dict, subchunks_node: dict, DC_node: dict):
        super().__init__()

        # ---- 预处理 ----
        self._preprocess_features(nodes, edges, subchunks_node, DC_node)
        # 维度定义（内部固定）
        edge_in_dim = self.edge_feat.shape[1]  # [log1p(deg(u)), log1p(mean(u)), type0(u), type1(u), log1p(deg(v)), log1p(mean(v)), type0(v), type1(v), log1p(tx), log1p(prop)]
        sub_in_dim = self.sub_feat.shape[1]  # [pos, log1p(deg(src)), log1p(mean(src)), type0(src), type1(src)]
        hidden_dim = 128

        self.edge_encoder = EdgeEncoder(edge_in_dim, hidden_dim)
        self.sub_encoder  = SubchunkEncoder(sub_in_dim, hidden_dim)
        self.graph_tf     = GraphTransformer(hidden_dim, num_heads=4, num_layers=2)
        self.sparse_attention = SparseAttentionTransformer(hidden_dim, num_heads=4, num_layers=4)
        self.value_head   = ValueHead(hidden_dim)

        self.S = self.sub_feat.size(0)
        self.E = self.edge_feat.size(0)
        self.to(device)

    def forward(self):
        # Encode raw features
        edge_repr = self.edge_encoder(self.edge_feat)   # [E, D]
        sub_repr  = self.sub_encoder(self.sub_feat)     # [S, D]
        # print(edge_repr.shape)

        # Graph transformer fusion (only process edges)
        refined_edge = self.graph_tf(edge_repr, attn_mask=self.edge_attn_mask)  # [E, D]
        refined_sub = sub_repr                  # [S, D]

        # Create one-hot indicator vectors
        S = refined_sub.shape[0]
        E = refined_edge.shape[0]
        device_ = refined_sub.device
        indicator_sub = torch.zeros((S, 2), dtype=refined_sub.dtype, device=device_)
        indicator_sub[:, 0] = 1.0  # [1,0] for subchunks
        indicator_edge = torch.zeros((E, 2), dtype=refined_edge.dtype, device=device_)
        indicator_edge[:, 1] = 1.0  # [0,1] for edges

        # Concatenate indicator to features
        refined_sub_ = torch.cat([refined_sub, indicator_sub], dim=1)   # [S, D+2]
        refined_edge_ = torch.cat([refined_edge, indicator_edge], dim=1) # [E, D+2]

        # --- Integrate SparseAttentionTransformer and ValueHead ---
        # (Assume SparseAttentionTransformer returns updated sub/edge representations)
        refined_sub_attn, refined_edge_attn = self.sparse_attention(
            refined_edge_,  refined_sub_, attn_mask=self.sparse_attn_mask
        )
        # Critic: pool token representations and output a scalar V(s)
        value = self.value_head(refined_sub_attn, refined_edge_attn)  # [1]
        return value

    def _preprocess_features(self, nodes, edges, subchunks_node, DC_node):
        import torch

        node_type_map = {"switch": 0, "SWITCH": 0, "Switch": 0,
                         "gpu": 1, "GPU": 1, "Gpu": 1}
        node_ids = sorted(nodes.keys())

        deg, mean_deg, ntype = {}, {}, {}
        for nid in node_ids:
            feat = nodes[nid]
            deg[nid] = float(feat["degree"])
            mean_deg[nid] = float(feat["mean_neighbor_degree"])
            t = feat["node_type"]
            ntype[nid] = int(t) if isinstance(t, int) else node_type_map.get(str(t), 0)

        def node_feat_vec_onehot(nid: int):
            # [log1p(deg), log1p(mean_deg), one-hot type (2-dim)]
            d = torch.log1p(torch.tensor(deg[nid], dtype=torch.float32))
            md = torch.log1p(torch.tensor(mean_deg[nid], dtype=torch.float32))
            t_idx = ntype[nid]
            t_onehot = torch.zeros(2, dtype=torch.float32)
            t_onehot[t_idx] = 1.0
            return torch.stack([d, md, t_onehot[0], t_onehot[1]])

        # --- Edge features ---
        edge_feat_list, uv = [], []
        for eid in edges.keys():
            u, v = int(edges[eid]["u"]), int(edges[eid]["v"])
            tx, pr = float(edges[eid]["tx_lat"]), float(edges[eid]["prop_lat"])
            tx_log, pr_log = torch.log1p(torch.tensor(tx, dtype=torch.float32)), torch.log1p(
                torch.tensor(pr, dtype=torch.float32))
            u_vec, v_vec = node_feat_vec_onehot(u), node_feat_vec_onehot(v)
            feat_vec = torch.cat([u_vec, v_vec, tx_log.unsqueeze(0), pr_log.unsqueeze(0)])  # [10]
            edge_feat_list.append(feat_vec)
            uv.append([u, v])

        self.edge_feat = torch.stack(edge_feat_list, dim=0).to(device)
        self.edge_index_uv = torch.tensor(uv, dtype=torch.long, device=device)

        # --- Subchunk features ---
        sub_feat_list, src_nodes = [], []
        for scid, sc in subchunks_node.items():
            src, pos = int(sc["source_node"]), float(sc["pos_in_node"])
            src_vec = node_feat_vec_onehot(src)
            feat_vec = torch.cat([torch.tensor([pos], dtype=torch.float32), src_vec])  # [5]
            sub_feat_list.append(feat_vec)
            src_nodes.append(src)

        if sub_feat_list:
            self.sub_feat = torch.stack(sub_feat_list, dim=0).to(device)
        else:
            self.sub_feat = torch.zeros((0, 5), dtype=torch.float32, device=device)

        self.src_nodes = torch.tensor(src_nodes, dtype=torch.long, device=device)

        # --- Global standardization (optional, improves stability) ---
        def normalize_features(feat_tensor: torch.Tensor):
            if feat_tensor.numel() == 0:
                return feat_tensor
            mean = feat_tensor.mean(dim=0, keepdim=True)
            std = feat_tensor.std(dim=0, keepdim=True) + 1e-6
            return (feat_tensor - mean) / std

        self.edge_feat = normalize_features(self.edge_feat)
        self.sub_feat = normalize_features(self.sub_feat)

        # --- Build attention masks ---
        edge_uv = [(int(edges[eid]["u"]), int(edges[eid]["v"])) for eid in edges]
        S, E = len(subchunks_node), len(edge_uv)

        edge_attn_mask = torch.zeros((E, E), dtype=torch.float32, device=device)
        sparse_attn_mask = torch.zeros((E + S, E + S), dtype=torch.float32, device=device)

        # Edge ↔ Edge
        for i, (u_i, v_i) in enumerate(edge_uv):
            for j, (u_j, v_j) in enumerate(edge_uv):
                if not (i == j or v_i == u_j):
                    edge_attn_mask[i, j] = float('-inf')
                    sparse_attn_mask[i, j] = float('-inf')

        # Subchunk → Edge
        for sid, sc in subchunks_node.items():
            src = sc["source_node"]
            for i, (u, v) in enumerate(edge_uv):
                if v == src:  # 禁止回到源节点
                    sparse_attn_mask[E + sid, i] = float('-inf')

        # Edge → Subchunk
        for i, (u, v) in enumerate(edge_uv):
            for sid, sc in subchunks_node.items():
                src = sc["source_node"]
                if v == src:  # 禁止回到 subchunk 的源节点
                    sparse_attn_mask[i, E + sid] = float('-inf')

        # Subchunk ↔ Subchunk 默认全 0.0，不需改

        # node → DC 映射
        node2dc = {}
        for dc_id, nodes in DC_node.items():
            for n in nodes:
                node2dc[n] = dc_id

        # DC 约束：禁止跨 DC 把 subchunk 送回源 DC
        for i, (u, v) in enumerate(edge_uv):
            for sid, sc in subchunks_node.items():
                src = sc["source_node"]
                dc_u, dc_v, dc_src = node2dc.get(u, -1), node2dc.get(v, -1), node2dc.get(src, -1)
                if dc_u == -1 or dc_v == -1 or dc_src == -1:
                    print("ERROR!")
                if dc_u != dc_v and dc_v == dc_src:
                    sparse_attn_mask[i, E + sid] = float('-inf')
                    sparse_attn_mask[E + sid, i] = float('-inf')

        self.edge_attn_mask = edge_attn_mask
        self.sparse_attn_mask = sparse_attn_mask




# ---------- Sparse Attention Graph Transformer ----------
class SparseAttentionTransformer(nn.Module):
    """
    Sparse attention transformer for graph-based representations.
    Processes the concatenated sequence of subchunk and edge tokens with indicator.
    Consumes an externally provided attention mask (attn_mask) for flexible sparse attention.
    The mask should be a float tensor of shape [S+E, S+E], with 0.0 for allowed and -inf for disallowed attention.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.proj_in = nn.Linear(hidden_dim + 2, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, edge_repr: torch.Tensor, sub_repr: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        sub_repr: [S, D+2], edge_repr: [E, D+2]
        attn_mask: Optional[Tensor] of shape [S+E, S+E], float, 0.0 for allowed, -inf for disallowed.
        Returns updated (sub_repr, edge_repr) after cross attention.
        The attention mask must be provided externally.
        """
        S = sub_repr.size(0)
        E = edge_repr.size(0)
        # Concatenate sub and edge tokens
        tokens = torch.cat([edge_repr,sub_repr], dim=0)   # [S+E, D+2]
        # Project to hidden_dim
        x = self.proj_in(tokens).unsqueeze(0)  # [1, S+E, hidden_dim]
        # Pass the externally provided mask to TransformerEncoder
        x_out = self.encoder(x, mask=attn_mask)  # [1, S+E, hidden_dim]
        x_out = x_out.squeeze(0) # [S+E, hidden_dim]
        # Split back
        updated_edge = x_out[:E, :]
        updated_sub = x_out[E:, :]
        return updated_sub, updated_edge