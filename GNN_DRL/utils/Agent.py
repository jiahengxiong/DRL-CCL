import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import softmax

# ---- Safe normalization to avoid NaN when vector is zero ----
def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

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
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.LayerNorm(hidden_dim),
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
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.LayerNorm(hidden_dim),
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
            dropout=0.1,
            batch_first=True,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pre_ln = nn.LayerNorm(hidden_dim)
        self.post_ln = nn.LayerNorm(hidden_dim)


    def forward(self, edge_repr: torch.Tensor, attn_mask=None):
        # edge_repr: [E, D]
        x = self.pre_ln(edge_repr).unsqueeze(0)  # [1, E, D]
        x_out = self.encoder(x, mask=attn_mask)  # [1, E, D]
        x_out = self.post_ln(x_out.squeeze(0))   # [E, D]
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


# --------- Hard-Concrete activation (0/1 with STE) ----------
class HardConcrete(nn.Module):
    """
    Turn arbitrary logits into [0,1] values. In forward it can produce strict 0/1
    (via clamp), while backward uses a straight-through estimator so the network
    still receives gradients through the unclamped variable.
    This keeps everything *inside* the model instead of a downstream post-process.
    """
    def __init__(self, tau: float = 0.4, gamma: float = -0.05, zeta: float = 1.05):
        super().__init__()
        self.tau = tau
        self.gamma = gamma
        self.zeta = zeta
        # Optional: allow a simple programmatic anneal of tau (temperature)
        # Call self.set_tau(x) from your training loop if you want; otherwise, leave it.

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Map logits -> [0,1] with true 0/1 in forward; STE in backward."""
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1-1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1-u) + logits) / self.tau)
        else:
            s = torch.sigmoid(logits / self.tau)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        y_hard = s_bar.clamp(0.0, 1.0)
        y = y_hard + (s_bar - s_bar.detach())
        return y



from entmax import entmax15  # pip install entmax

# ---------- Scoring Head（输出稠密矩阵，激活=entmax15） ----------
class ScoringHead(nn.Module):
    """
    输出 [S, E] 权重矩阵；对每个 subchunk（每一行）做 entmax15 分布（天然稀疏，部分严格为 0）。
    """
    def __init__(self, hidden_dim: int, proj_dim: int = 64, scale: float = 5.0):
        super().__init__()
        self.proj_sub  = nn.Linear(hidden_dim, proj_dim, bias=True)
        self.proj_edge = nn.Linear(hidden_dim, proj_dim, bias=True)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(scale))))

        nn.init.xavier_uniform_(self.proj_sub.weight);  nn.init.zeros_(self.proj_sub.bias)
        nn.init.xavier_uniform_(self.proj_edge.weight); nn.init.zeros_(self.proj_edge.bias)

    def forward(self, edge_repr: torch.Tensor, sub_repr: torch.Tensor) -> torch.Tensor:
        # sub_repr: [S, D], edge_repr: [E, D]
        sub_proj  = F.normalize(self.proj_sub(sub_repr),  dim=-1)   # [S, P]
        edge_proj = F.normalize(self.proj_edge(edge_repr), dim=-1)  # [E, P]

        # 打分 [S, E]
        scale  = self.log_scale.exp().clamp(0.5, 20.0)
        scores = (sub_proj @ edge_proj.t()) * scale

        # 直接用 entmax15 产生稀疏概率分布
        weights = entmax15(scores, dim=-1)  # [S, E]

        return weights

class AttnPool1D(nn.Module):
    """单头注意力池化：对一组 [N, D] token 做加权和，权重来自可学习的查询。"""
    def __init__(self, d_in, d_hid):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hid),
            nn.Tanh(),
            nn.Linear(d_hid, 1)
        )

    def forward(self, x):  # x:[N,D]
        if x.numel() == 0:
            return torch.zeros(1, x.size(-1), device=x.device, dtype=x.dtype)
        score = self.proj(x)                    # [N,1]
        alpha = torch.softmax(score, dim=0)     # [N,1]
        return (alpha * x).sum(dim=0, keepdim=True)  # [1,D]


class ValueHead(nn.Module):
    """
    Stronger critic head.
    - For sub/edge token sets, we compute three pools: attention, mean, max.
    - Project each pooled triple down to hidden_dim.
    - Combine sub/edge embeddings with |
      difference| and elementwise product to capture interactions.
    - A deeper MLP predicts a scalar V(s).
    Keeps the same signature: forward(sub_repr, edge_repr) -> [1].
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Pools
        self.pool_sub_attn  = AttnPool1D(hidden_dim, hidden_dim // 2)
        self.pool_edge_attn = AttnPool1D(hidden_dim, hidden_dim // 2)

        # Projections for pooled triples (attn, mean, max) -> hidden_dim
        self.ln_sub  = nn.LayerNorm(3 * hidden_dim)
        self.ln_edge = nn.LayerNorm(3 * hidden_dim)
        self.proj_sub = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.proj_edge = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )

        # Final predictor on [sub_emb, edge_emb, |sub-edge|, sub*edge] (4*hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2 * hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Init
        for mod in [self.proj_sub[0], self.proj_edge[0], self.mlp[1], self.mlp[4], self.mlp[6]]:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, sub_repr: torch.Tensor, edge_repr: torch.Tensor) -> torch.Tensor:
        D = sub_repr.size(-1) if sub_repr.numel() > 0 else edge_repr.size(-1)
        device = sub_repr.device if sub_repr.numel() > 0 else edge_repr.device

        def safe_stats(x: torch.Tensor):
            if x.numel() == 0:
                zeros = torch.zeros(1, D, device=device, dtype=edge_repr.dtype)
                return zeros, zeros, zeros
            attn = self.pool_sub_attn(x) if x is sub_repr else self.pool_edge_attn(x)
            mean = x.mean(dim=0, keepdim=True)
            maxv = x.max(dim=0, keepdim=True).values
            return attn, mean, maxv

        # sub pools
        sub_attn, sub_mean, sub_max = safe_stats(sub_repr)
        sub_triple = torch.cat([sub_attn, sub_mean, sub_max], dim=-1)
        sub_emb = self.proj_sub(self.ln_sub(sub_triple))  # [1, D]

        # edge pools
        edge_attn, edge_mean, edge_max = safe_stats(edge_repr)
        edge_triple = torch.cat([edge_attn, edge_mean, edge_max], dim=-1)
        edge_emb = self.proj_edge(self.ln_edge(edge_triple))  # [1, D]

        # interactions
        diff = torch.abs(sub_emb - edge_emb)
        prod = sub_emb * edge_emb
        feat = torch.cat([sub_emb, edge_emb, diff, prod], dim=-1)  # [1, 4D]

        v = self.mlp(feat).squeeze(-1)  # [1]
        return v


# ---------- Agent ----------
class AllGatherAgent(nn.Module):
    """
    __init__(nodes, edges, subchunks_node):
      - 三份字典直接进来；内部预处理为张量
    forward():
      - 编码 → 图变换 → 稀疏注意力融合 → 打分 → 输出稠密 [S,E] 权重矩阵；并通过 ValueHead 池化得到标量 V(s)
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
        self.scoring      = ScoringHead(hidden_dim, proj_dim=64)
        self.value_head  = ValueHead(hidden_dim)

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


        # --- Integrate SparseAttentionTransformer and ScoringHead ---
        # (Assume SparseAttentionTransformer returns updated sub/edge representations)
        refined_sub_attn, refined_edge_attn = self.sparse_attention(
         refined_edge_,  refined_sub_, attn_mask=self.sparse_attn_mask
        )
        # Use ScoringHead to produce weights, and ValueHead to produce V(s)
        weights_dense = self.scoring(refined_edge_attn, refined_sub_attn)  # [S,E]
        value = self.value_head(refined_sub_attn, refined_edge_attn)       # [1]
        return weights_dense, value

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
            dropout=0.1,
            batch_first=True,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pre_ln = nn.LayerNorm(hidden_dim)
        self.post_ln = nn.LayerNorm(hidden_dim)

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
        tokens = torch.cat([edge_repr, sub_repr], dim=0)   # [S+E, D+2]
        # Project and stabilize
        x = self.pre_ln(self.proj_in(tokens)).unsqueeze(0)  # [1, S+E, hidden_dim]
        # Pass the externally provided mask to TransformerEncoder
        x_out = self.encoder(x, mask=attn_mask)             # [1, S+E, hidden_dim]
        x_out = self.post_ln(x_out.squeeze(0))              # [S+E, hidden_dim]
        # Split back
        updated_edge = x_out[:E, :]
        updated_sub = x_out[E:, :]
        return updated_sub, updated_edge


# ===================== RewardNet (shape-agnostic, Transformer + SE) =====================
class RewardNet(nn.Module):
    """
    输入：agent 的静态特征张量 + SE[S,E]
    流程：
      sub/edge 编码 -> 用 SE 做双向加权聚合 (形状自适应) -> 拼接指示位 -> 稀疏跨注意力 -> 池化 -> 标量 \hat R
    """
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 2,
                 fuse_dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 复用你已有的 encoder / transformer 类型
        self.sub_encoder  = SubchunkEncoder(in_dim=5,  hidden_dim=hidden_dim)
        self.edge_encoder = EdgeEncoder(   in_dim=10, hidden_dim=hidden_dim)
        self.graph_tf     = GraphTransformer(hidden_dim, num_heads=num_heads, num_layers=num_layers)

        # 融合 (静态表征 || 通过 SE 聚合得到的互视表征) -> 回到 D
        self.fuse_sub = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fuse_dropout),
        )
        self.fuse_edge = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fuse_dropout),
        )

        # 稀疏跨模态注意力（复用你的 SparseAttentionTransformer，输入会多 2 维指示位）
        self.sparse_attn = SparseAttentionTransformer(hidden_dim, num_heads=num_heads, num_layers=num_layers)

        # 池化 + 回归头
        self.pool_sub_attn  = AttnPool1D(hidden_dim, hidden_dim // 2)
        self.pool_edge_attn = AttnPool1D(hidden_dim, hidden_dim // 2)

        self.proj_sub  = nn.Linear(3 * hidden_dim, hidden_dim)
        self.proj_edge = nn.Linear(3 * hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.proj_sub.weight);  nn.init.zeros_(self.proj_sub.bias)
        nn.init.xavier_uniform_(self.proj_edge.weight); nn.init.zeros_(self.proj_edge.bias)

        self.out_head = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2 * hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def _row_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # 对每一行做和为 1 的归一化（支持非负/稀疏，给个 eps 防 0）
        denom = x.sum(dim=-1, keepdim=True).clamp_min(eps)
        return x / denom

    @staticmethod
    def _col_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # 对每一列做和为 1 的归一化
        denom = x.sum(dim=0, keepdim=True).clamp_min(eps)
        return x / denom

    def forward(self, agent: "AllGatherAgent", se_weights: torch.Tensor) -> torch.Tensor:
        """
        se_weights: [S, E]
        return: 标量 \hat R（未经缩放）
        """
        assert se_weights.dim() == 2, "SE must be [S, E]"
        S, E = se_weights.shape
        device = se_weights.device

        # 1) 编码静态特征
        sub_repr  = self.sub_encoder(agent.sub_feat)                 # [S, D]
        edge_repr = self.edge_encoder(agent.edge_feat)               # [E, D]
        edge_repr = self.graph_tf(edge_repr, agent.edge_attn_mask)   # [E, D]

        # 2) 用 SE 做双向加权聚合（形状自适应，无需固定 Linear 维度）
        #   对每个 sub： se_row_norm[s,:] 作为对 edge_repr 的权重 -> 得到 se_sub_tok[s]
        se_row_norm = self._row_norm(se_weights).to(device)          # [S, E]
        se_sub_tok  = se_row_norm @ edge_repr                        # [S, D]

        #   对每条 edge： se_col_norm[:,e] 作为对 sub_repr 的权重 -> 得到 se_edge_tok[e]
        se_col_norm = self._col_norm(se_weights).to(device)          # [S, E]
        se_edge_tok = se_col_norm.t() @ sub_repr                     # [E, D]

        # 3) 拼接融合
        sub_fused  = self.fuse_sub(torch.cat([sub_repr,  se_sub_tok],  dim=-1))   # [S, D]
        edge_fused = self.fuse_edge(torch.cat([edge_repr, se_edge_tok], dim=-1))  # [E, D]

        # 4) 加 2 维指示位，做稀疏跨注意力
        indicator_sub = torch.zeros((S, 2), dtype=sub_fused.dtype, device=device);  indicator_sub[:, 0] = 1.0
        indicator_edge = torch.zeros((E, 2), dtype=edge_fused.dtype, device=device); indicator_edge[:, 1] = 1.0

        sub_tok  = torch.cat([sub_fused,  indicator_sub],  dim=-1)   # [S, D+2]
        edge_tok = torch.cat([edge_fused, indicator_edge], dim=-1)   # [E, D+2]

        sub_upd, edge_upd = self.sparse_attn(edge_tok, sub_tok, attn_mask=agent.sparse_attn_mask)  # [S,D],[E,D]

        # 5) 池化成全局图表征
        def pools(x, pool_attn):
            if x.numel() == 0:
                z = torch.zeros(1, self.hidden_dim, device=device, dtype=x.dtype)
                return z, z, z
            a = pool_attn(x)                   # [1,D]
            m = x.mean(0, keepdim=True)        # [1,D]
            M = x.max(0, keepdim=True).values  # [1,D]
            return a, m, M

        a_sub, m_sub, M_sub   = pools(sub_upd,  self.pool_sub_attn)
        a_edge, m_edge, M_edge = pools(edge_upd, self.pool_edge_attn)

        sub_emb  = self.proj_sub(torch.cat([a_sub,  m_sub,  M_sub ], dim=-1))   # [1,D]
        edge_emb = self.proj_edge(torch.cat([a_edge, m_edge, M_edge], dim=-1))  # [1,D]

        feat = torch.cat([sub_emb, edge_emb, torch.abs(sub_emb - edge_emb), sub_emb * edge_emb], dim=-1)  # [1,4D]
        hat_R = self.out_head(feat).squeeze(-1).squeeze(0)  # 标量
        return hat_R

