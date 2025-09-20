from decimal import Decimal
import math
import config
from collections import defaultdict

from collections import defaultdict

def build_subchunk_weights_from_policy(policy, eps: float = 1e-12, normalize: bool = True):
    """
    将 policy（形如 [sub_id, u, v, send_time] 的记录列表）转成稀疏优先级表：
        mats[sub_id][(u, v)] = 1.0 / (send_time + eps)

    - 若某 subchunk 从未在链路 (u,v) 上发送，则 mats[sub_id] 中无该键。
    - 同一 (sub_id, u, v) 出现多次，保留更大的权重（即更早的 send_time）。
    - 归一化逻辑（normalize=True）：
        * 默认：跨所有 subchunk，对每条 (u,v) 分别归一化；
        * 即 link (u,v) 上所有 subchunk 的最大权重 = 1.0。
    """
    mats = {}
    # 先收集权重
    for rec in policy:
        if len(rec) < 4:
            raise ValueError(f"Bad policy item (need 4 fields): {rec}")
        sid, u, v, t = rec
        if not isinstance(u, int) or not isinstance(v, int):
            raise TypeError(f"Nodes must be int indices: got u={u}, v={v}")
        if t is None or float(t) < 0:
            continue
        w = 1.0 / (float(t) + eps)
        bucket = mats.setdefault(sid, {})
        key = (u, v)
        if bucket.get(key, 0.0) < w:
            bucket[key] = w

    if normalize:
        # 先找每条 (u,v) 跨 subchunk 的最大权重
        link_max = {}
        for sid, d in mats.items():
            for (u, v), w in d.items():
                link_max[(u, v)] = max(link_max.get((u, v), 0.0), w)

        # 再归一化
        for sid, d in mats.items():
            for (u, v), w in d.items():
                m = link_max.get((u, v), 1.0)
                if m > 0:
                    d[(u, v)] = float(w / m)
                else:
                    d[(u, v)] = 0.0
    return mats
# ========= Packet 级 allgather 仿真（严格按 link 队列优先级 + FIFO；权重<=0 不入队；BR cut-through；sub_id从0开始）=========
worst_reward_so_far = 0.0  # global tracker for worst reward (most negative)

def simulate_allgather_pipeline_bfs(
    G,
    packet_size_per_subchunk: Decimal,
    gpu_nodes,
    trees_override=None, depths_override=None, max_depths_override=None,  # 兼容占位，不使用
    verbose=False,
    subchunk_priority_mats=None,   # { sub_id: 2D矩阵(list/tuple/ndarray) 或 {(u,v): weight} }
    is_border_router_fn=None,      # 新增：可选回调，判定节点是否为边界路由器（BR）
):
    """
    极简版：
      1) 流水线：链路空闲就发，忙则排队。
      2) GPU store-forward：在 recv_end 时把该 subchunk 入队到所有出边（仅对权重>0 的边；已收过的不再入队）。
      3) 边界路由器（BR）cut-through：在 recv_start 时入队。
      4) 当一个 (u,v) 的队列中加入新项时，立刻对队列按 (-priority, FIFO) 排序；优先级越大越先发；同权 FIFO。
      5) 打印时严格按“发送开始时间”排序输出日志。
    """
    from decimal import Decimal as _Dec
    from collections import defaultdict
    import heapq

    # ---------- 新增: eligibility/wait/used/rank 追踪 ----------
    elig_time = defaultdict(lambda: None)   # (node, sid) -> Decimal
    used_edges = set()
    enqueue_meta = {}

    # ---------- 日志 ----------
    VERBOSE = bool(verbose)
    def log(msg: str):
        if VERBOSE:
            print(msg)

    # ---------- subchunk 编号（从 0 开始连续） ----------
    try:
        num_subchunks_per_gpu = int(config.num_chunk)
    except Exception:
        num_subchunks_per_gpu = 1

    subchunks = []
    sub_src_of = {}
    for gi, src in enumerate(gpu_nodes):
        for k in range(num_subchunks_per_gpu):
            sid = gi * num_subchunks_per_gpu + k
            subchunks.append(sid)
            sub_src_of[sid] = src

    # ---------- 边界路由器判断（仅看节点属性，或可选回调） ----------
    def is_br(node) -> bool:
        t = G.nodes[node].get('type', None)
        if isinstance(t, str) and t.lower() in {"br", "border", "border_router", "border-router", "switch"}:
            return True
        if isinstance(t, int) and t == 1:
            return True
        if is_border_router_fn is not None:
            try:
                return bool(is_border_router_fn(G, node))
            except Exception:
                pass
        return False

    # ---------- 取权重 ----------
    def get_weight(sub_id, u, v):
        if subchunk_priority_mats is None:
            return 0.0
        mat = subchunk_priority_mats.get(sub_id)
        if mat is None:
            return 0.0
        if isinstance(mat, dict):
            return float(mat.get((u, v), 0.0))
        # 矩阵型
        try:
            return float(mat[u][v])
        except Exception:
            return 0.0

    # ---------- 发送/传播时间 ----------
    def link_ser_prop(u, v):
        e = G.edges[(u, v)]
        # Prefer directly provided per-subchunk transmission latency (seconds).
        ser = e.get('transmission_latency', None)
        if ser is None:
            # Fallback: compute serialization time from capacity if available.
            cap = e.get('link_capacity', None)  # bytes / s (Decimal)
            if cap is None:
                raise KeyError(f"Edge {(u, v)} missing 'transmission_latency' and 'link_capacity'")
            ser = packet_size_per_subchunk / cap  # s (Decimal)
        # Propagation latency; default to 0 if not present.
        prop = e.get('propagation_latency', _Dec(0))  # s (Decimal)
        return ser, prop

    # ---------- 状态 ----------
    # 每条链路的队列（list），元素：{'sid', 'pri', 'enq_time', 'aseq'}
    edge_q = defaultdict(list)
    link_free_time = defaultdict(lambda: _Dec(0))
    have_time = defaultdict(lambda: None)  # (node, sid) -> Decimal 完全接收时刻
    enqueued = set()  # 防止重复入队： (sid,u,v)
    aseq_counter = 0  # FIFO 序号
    send_logs = []    # (t_start, t_end, t_recv_start, t_recv_end, u, v, sid)

    # ---------- 事件 ----------
    event_q = []  # (time, seq, kind, data)
    eid = 0
    def schedule(t, kind, **data):
        nonlocal eid
        heapq.heappush(event_q, (t, eid, kind, data))
        eid += 1

    # ---------- 入队 + 排序 + 如果空闲则尝试发送 ----------
    def enqueue(u, v, sid, now: _Dec, *, force: bool = False, pri_override: float = None):
        nonlocal aseq_counter
        # Prevent duplicate enqueue for the same (sid, u, v)
        if (sid, u, v) in enqueued:
            return  # already enqueued this subchunk on this link
        w = get_weight(sid, u, v)
        if w <= 0.0 and not force:
            return
        pri = float(w) if not force else (float(pri_override) if pri_override is not None else 1e-6)
        enqueued.add((sid, u, v))
        item = {'sid': sid, 'pri': pri, 'enq_time': now, 'aseq': aseq_counter}
        aseq_counter += 1
        q = edge_q[(u, v)]
        q.append(item)
        # 立刻按 (-priority, aseq) 排序
        q.sort(key=lambda x: (-x['pri'], x['aseq']))
        # 记录 rank, qlen, 入队时刻
        rank = 1 + next(i for i,x in enumerate(q) if x is item)
        enqueue_meta[(sid,u,v)] = {'rank_enq': rank, 'qlen_enq': len(q), 't_enq': now}
        try_start_next(u, v, now)

    # ---------- 原始: 仅对权重>0的出边入队 ----------

    # ---------- 尝试在 (u,v) 上启动下一次发送 ----------
    def try_start_next(u, v, now: _Dec):
        q = edge_q[(u, v)]
        if not q:
            return
        t_free = link_free_time[(u, v)]
        if now < t_free:
            return  # 链路忙，等 send_end 触发
        # 队头就绪（入队即就绪），发之
        item = q.pop(0)
        sid = item['sid']
        ser, prop = link_ser_prop(u, v)
        t_start = t_free if t_free > now else now
        t_end = t_start + ser
        t_recv_start = t_start + prop
        t_recv_end = t_start + ser + prop
        link_free_time[(u, v)] = t_end

        # 新增: used_edges, 等待时间
        used_edges.add((sid,u,v))
        t_elig = elig_time.get((u, sid), None)
        wait = (t_start - t_elig) if (t_elig is not None) else None

        schedule(t_end,        "send_end",   u=u, v=v, sid=sid)
        schedule(t_recv_start, "recv_start", u=u, v=v, sid=sid)
        schedule(t_recv_end,   "recv_end",   u=u, v=v, sid=sid)

        if VERBOSE:
            send_logs.append((t_start, t_end, t_recv_start, t_recv_end, u, v, sid))

    # ---------- 当节点获得 subchunk 时，按节点类型决定入队时机 ----------
    def on_recv_start(u, v, sid, now: _Dec):
        # 对于 BR，recv_start 立即把 sid 往所有出边入队（权重>0；且目标节点尚未收到）
        if is_br(v):
            # 新增: BR cut-through 时 eligibility
            elig_time[(v, sid)] = now
            # revert to original: only enqueue to edges with weight>0
            for w_node in G.successors(v):
                if get_weight(sid, v, w_node) > 0.0:
                    enqueue(v, w_node, sid, now)

    def on_recv_end(u, v, sid, now: _Dec):
        if have_time[(v, sid)] is None:
            have_time[(v, sid)] = now
            # 新增: GPU store-forward eligibility
            elig_time[(v, sid)] = now
            # 对于非 BR（GPU），store-forward：recv_end 再把 sid 往所有出边入队
            if not is_br(v):
                for w_node in G.successors(v):
                    if get_weight(sid, v, w_node) > 0.0:
                        enqueue(v, w_node, sid, now)

    # ---------- 初始化：源 GPU 自身在 t=0 已拥有自己的 subchunk ----------
    for sid in subchunks:
        src = sub_src_of[sid]
        have_time[(src, sid)] = _Dec(0)
        elig_time[(src, sid)] = _Dec(0)
        # 源节点按节点类型立即/稍后入队？——源是 GPU，等价于“已经完成接收”，因此 t=0 即入队
        for w_node in G.successors(src):
            if get_weight(sid, src, w_node) > 0.0:
                enqueue(src, w_node, sid, _Dec(0))

    # ---------- 事件循环 ----------
    while event_q:
        t, _, kind, data = heapq.heappop(event_q)
        if kind == "send_end":
            u, v, sid = data['u'], data['v'], data['sid']
            # 链路空出来了，立刻尝试继续发队头
            try_start_next(u, v, t)

        elif kind == "recv_start":
            u, v, sid = data['u'], data['v'], data['sid']
            on_recv_start(u, v, sid, t)

        elif kind == "recv_end":
            u, v, sid = data['u'], data['v'], data['sid']
            on_recv_end(u, v, sid, t)

        else:
            raise ValueError(f"Unknown event kind: {kind}")

    # ---------- 打印发送日志（按发送开始时间排序） ----------
    if VERBOSE and send_logs:
        for t_start, t_end, t_recv_start, t_recv_end, u, v, sid in sorted(send_logs, key=lambda x: (x[0], x[6] if len(x) > 6 else 0, x[4], x[5])):
            log(
                f"在 {float(t_start)*1e6:.3f} us，node {u} 发送 subchunk {sid} 到 node {v}；"
                f"在 {float(t_end)*1e6:.3f} us 发送完；"
                f"在 {float(t_recv_start)*1e6:.3f} us 开始接收；"
                f"在 {float(t_recv_end)*1e6:.3f} us 接收完"
            )

    # ---------- 统计 makespan（仅 GPU 节点） ----------
    per_sub_max = {}
    for sid in subchunks:
        mx = _Dec(0)
        for n in gpu_nodes:
            t = have_time[(n, sid)]
            if t is None:
                mx = _Dec('Infinity')
                break
            if t > mx:
                mx = t
        per_sub_max[sid] = mx

    makespan = max(per_sub_max.values()) if per_sub_max else _Dec(0)

    if VERBOSE:
        log("\n===== Summary =====")
        for s in sorted(per_sub_max):
            log(f"subchunk {s} fully delivered by time {float(per_sub_max[s])*1e6:.3f} us")
        log(f"TOTAL makespan: {float(makespan)*1e6:.3f} us")
    # print(f"TOTAL makespan: {float(makespan)*1e6:.3f} us")

    # ---------- 特征提取 ----------
    # node_features: {node_id: {"type": str, "is_br": int, "in_degree": int, "out_degree": int, ...}}
    # edge_features: {(u,v): {"prop_delay": float, "tx_delay": float, ...}}
    # subchunk_edge_features: {(sid,u,v): {...}}
    features = {}
    node_features = {}
    for n in G.nodes:
        ntype = G.nodes[n].get("type", None)
        ntype_str = "" if ntype is None else str(ntype)
        is_br_val = 1 if is_br(n) else 0
        in_deg = G.in_degree[n] if hasattr(G, "in_degree") else len(list(G.predecessors(n)))
        out_deg = G.out_degree[n] if hasattr(G, "out_degree") else len(list(G.successors(n)))
        node_features[n] = {
            "type": ntype_str,
            "is_br": is_br_val,
            "in_degree": int(in_deg),
            "out_degree": int(out_deg),
        }
    edge_features = {}
    for u, v in G.edges:
        e = G.edges[(u, v)]
        prop = float(e.get("propagation_latency", 0.0))
        tx = float(e.get("transmission_latency", 0.0))
        link_type = str(e.get("link_type", ""))
        edge_features[(u, v)] = {
            "prop_delay": prop,
            "tx_delay": tx,
            "link_type": link_type,
            "src_is_br": 1 if is_br(u) else 0,
            "dst_is_br": 1 if is_br(v) else 0,
            "src_is_gpu": 1 if (G.nodes[u].get("type","").lower()=="gpu") else 0,
            "dst_is_gpu": 1 if (G.nodes[v].get("type","").lower()=="gpu") else 0,
        }
    subchunk_edge_features = {}
    # send_logs: (t_start, t_end, t_recv_start, t_recv_end, u, v, sid)
    for log_entry in send_logs:
        t_start, t_end, t_recv_start, t_recv_end, u, v, sid = log_entry
        weight = get_weight(sid, u, v)
        t_elig = elig_time.get((u, sid), None)
        # used=1, wait_time, rank, qlen, etc.
        key = (sid, u, v)
        subchunk_edge_features[key] = {
            "weight_prev": float(weight),
            "used": 1,
            "t_eligible": float(t_elig) if t_elig is not None else None,
            "wait_time": float(t_start - t_elig) if t_elig is not None else None,
            "t_tx_start": float(t_start),
            "t_tx_end": float(t_end),
            "t_rx_start": float(t_recv_start),
            "t_rx_end": float(t_recv_end),
            "rank_at_enqueue": enqueue_meta.get((sid,u,v),{}).get('rank_enq'),
            "q_len_at_enqueue": enqueue_meta.get((sid,u,v),{}).get('qlen_enq'),
        }
    # 新增: 把没出现的 (sid,u,v) 都补全
    for sid in subchunks:
        for (u,v) in G.edges:
            if (sid,u,v) not in subchunk_edge_features:
                weight = get_weight(sid,u,v)
                t_elig = elig_time.get((u,sid), None)
                subchunk_edge_features[(sid,u,v)] = {
                    "weight_prev": float(weight),
                    "used": 0,
                    "t_eligible": float(t_elig) if t_elig is not None else None,
                    "wait_time": None,
                    "t_tx_start": None,
                    "t_tx_end": None,
                    "t_rx_start": None,
                    "t_rx_end": None,
                    "rank_at_enqueue": None,
                    "q_len_at_enqueue": None,
                }
    features["node_features"] = node_features
    features["edge_features"] = edge_features
    features["subchunk_edge_features"] = subchunk_edge_features

    # ========== 归一化逻辑 ==========
    # 1. 节点最大度数
    max_deg = 1
    for n in G.nodes:
        deg = G.degree[n] if hasattr(G, "degree") else (G.in_degree[n] + G.out_degree[n])
        if deg > max_deg:
            max_deg = deg
    # 2. 边最大传播/传输延迟
    max_prop = max([abs(float(e.get("propagation_latency", 0.0))) for e in G.edges.values()] + [1e-9])
    max_tx = max([abs(float(e.get("transmission_latency", 0.0))) for e in G.edges.values()] + [1e-9])
    # 3. subchunk完成最大时间
    max_makespan = float(makespan) if makespan is not None else 1.0
    # 4. subchunk-edge最大权重
    max_weight = max([abs(float(f['weight_prev'])) for f in subchunk_edge_features.values()] + [1e-9])
    # 5. subchunk-edge最大时间（用于归一化时间特征）
    max_t = max([
        abs(float(f[x])) for f in subchunk_edge_features.values()
        for x in ['t_tx_start','t_tx_end','t_rx_start','t_rx_end','t_eligible'] if f[x] is not None
    ] + [1e-9, max_makespan])
    # 6. subchunk-edge最大等待
    max_wait = max([
        abs(float(f['wait_time'])) for f in subchunk_edge_features.values() if f['wait_time'] is not None
    ] + [1e-9])
    # 7. subchunk-edge最大rank
    max_rank = max([
        abs(int(f['rank_at_enqueue'])) for f in subchunk_edge_features.values() if f['rank_at_enqueue'] is not None
    ] + [1])

    # ========== 视图 ==========
    # node_features_list: 每节点 [deg_norm, is_br]
    node_features_list = []
    for n in sorted(G.nodes):
        deg = G.degree[n] if hasattr(G, "degree") else (G.in_degree[n] + G.out_degree[n])
        deg_norm = float(deg) / max_deg if max_deg > 0 else 0.0
        is_br_val = 1 if is_br(n) else 0
        node_features_list.append([deg_norm, is_br_val])

    # edge_features_list: 每边 [prop_norm, tx_norm, onehot_type...]
    edge_features_list = []
    edge_index = []
    for (u, v) in G.edges:
        e = edge_features[(u, v)]
        prop = e['prop_delay']
        tx = e['tx_delay']
        prop_norm = float(prop) / max_prop if max_prop > 0 else 0.0
        tx_norm = float(tx) / max_tx if max_tx > 0 else 0.0
        # onehot: NVlink=[1,0], Switch=[0,1], else [0,0]
        link_type = G.edges[(u, v)].get('link_type', '')
        link_type_str = str(link_type).lower()
        if link_type_str == "nvlink":
            onehot = [1,0]
        elif link_type_str == "switch":
            onehot = [0,1]
        else:
            onehot = [0,0]
        edge_features_list.append([prop_norm, tx_norm] + onehot)
        edge_index.append((u, v))

    # subchunk_edge_features_list: 每 (sid,u,v) 的 [sid, u, v, weight_norm, t_tx_start_norm, t_tx_end_norm, t_rx_start_norm, t_rx_end_norm, wait_norm, elig_norm, rank_norm]
    subchunk_edge_features_list = []
    for (sid, u, v), f in subchunk_edge_features.items():
        weight = float(f['weight_prev'])
        weight_norm = weight / max_weight if max_weight > 0 else 0.0
        t_tx_start_norm = float(f['t_tx_start']) / max_t if f['t_tx_start'] is not None and max_t > 0 else 0.0
        t_tx_end_norm = float(f['t_tx_end']) / max_t if f['t_tx_end'] is not None and max_t > 0 else 0.0
        t_rx_start_norm = float(f['t_rx_start']) / max_t if f['t_rx_start'] is not None and max_t > 0 else 0.0
        t_rx_end_norm = float(f['t_rx_end']) / max_t if f['t_rx_end'] is not None and max_t > 0 else 0.0
        wait_norm = float(f['wait_time']) / max_wait if f['wait_time'] is not None and max_wait > 0 else 0.0
        elig_norm = float(f['t_eligible']) / max_t if f['t_eligible'] is not None and max_t > 0 else 0.0
        rank_norm = float(f['rank_at_enqueue']) / max_rank if f['rank_at_enqueue'] is not None and max_rank > 0 else 0.0
        subchunk_edge_features_list.append([
            sid, u, v,
            weight_norm,
            t_tx_start_norm, t_tx_end_norm, t_rx_start_norm, t_rx_end_norm,
            wait_norm, elig_norm, rank_norm
        ])

    features["node_features_list"] = node_features_list
    features["edge_features_list"] = edge_features_list
    features["subchunk_edge_features_list"] = subchunk_edge_features_list

    # 保持原有 list_view
    edge_static = [[
        edge_features[(u,v)]['prop_delay'],
        edge_features[(u,v)]['tx_delay'],
        edge_features[(u,v)]['src_is_br'],
        edge_features[(u,v)]['dst_is_br'],
        edge_features[(u,v)]['src_is_gpu'],
        edge_features[(u,v)]['dst_is_gpu'],
    ] for (u,v) in edge_index]
    node_static = [[
        node_features[n]['is_br'],
        node_features[n]['in_degree'],
        node_features[n]['out_degree'],
    ] for n in sorted(G.nodes)]
    subchunk_edge_list = [[
        sid,u,v,
        f['weight_prev'], f['used'],
        f['t_eligible'], f['wait_time'],
        f['t_tx_start'], f['t_tx_end'],
        f['t_rx_start'], f['t_rx_end'],
        f['rank_at_enqueue'], f['q_len_at_enqueue'],
    ] for ((sid,u,v),f) in subchunk_edge_features.items()]
    features['list_view'] = {
        'edge_index': edge_index,
        'edge_static': edge_static,
        'node_static': node_static,
        'subchunk_edge_list': subchunk_edge_list,
    }
    # ========== PyTorch Geometric 风格 =========
    features['pyg_edge_index'] = [[u for (u,v) in edge_index], [v for (u,v) in edge_index]]
    features['pyg_edge_attr'] = edge_features_list
    features['pyg_node_attr'] = node_features_list
    features['pyg_subchunk_edge_attr'] = subchunk_edge_features_list

    # ---------- Reward Calculation (Simplified, unfinished subchunks and makespan) ----------
    # After computing per_sub_max and makespan:
    # - Count how many subchunks are unfinished (float('inf'))
    # - Define completion_rate = finished / total
    # - If not all are finished, set reward = completion_rate in [0,1]
    # - If all are finished, set reward = 1.0 / (1.0 + float(makespan)) (smaller makespan gets higher reward)
    per_sub_values = list(per_sub_max.values()) if per_sub_max else []
    total = len(per_sub_values)
    # unfinished: those with infinite makespan
    unfinished = [
        v for v in per_sub_values
        if v == float("inf") or (hasattr(v, 'is_infinite') and v.is_infinite()) or (isinstance(v, float) and math.isinf(v))
    ]
    unfinished_count = len(unfinished)
    finished = total - unfinished_count
    completion_rate = finished / total if total > 0 else 0.0

    if unfinished_count > 0:
        reward = -30.0 * unfinished_count
    else:
        # 使用倒数型 reward: makespan 越小 reward 越大，且变化平滑
        makespan_us = float(makespan) * 1e6
        reward = (1e4 / (makespan_us + 1.0)) ** 3

    # print(f"[Sim] finished={finished}/{total}, unfinished={unfinished_count}, makespan={makespan}, reward={reward}")
    return reward