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
    verbose=True,
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
    def enqueue(u, v, sid, now: _Dec):
        nonlocal aseq_counter
        if have_time[(v, sid)] is not None:
            return
        if (sid, u, v) in enqueued:
            return
        w = get_weight(sid, u, v)
        if w <= 0.0:
            return
        enqueued.add((sid, u, v))
        item = {'sid': sid, 'pri': float(w), 'enq_time': now, 'aseq': aseq_counter}
        aseq_counter += 1
        q = edge_q[(u, v)]
        q.append(item)
        # 立刻按 (-priority, aseq) 排序
        q.sort(key=lambda x: (-x['pri'], x['aseq']))
        # 新增: 记录 rank, qlen, 入队时刻
        rank = 1 + next(i for i,x in enumerate(q) if x is item)
        enqueue_meta[(sid,u,v)] = {'rank_enq': rank, 'qlen_enq': len(q), 't_enq': now}
        try_start_next(u, v, now)

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
            for w in G.successors(v):
                if have_time[(w, sid)] is None:
                    enqueue(v, w, sid, now)

    def on_recv_end(u, v, sid, now: _Dec):
        if have_time[(v, sid)] is None:
            have_time[(v, sid)] = now
            # 新增: GPU store-forward eligibility
            elig_time[(v, sid)] = now
            # 对于非 BR（GPU），store-forward：recv_end 再把 sid 往所有出边入队
            if not is_br(v):
                for w in G.successors(v):
                    if have_time[(w, sid)] is None:
                        enqueue(v, w, sid, now)

    # ---------- 初始化：源 GPU 自身在 t=0 已拥有自己的 subchunk ----------
    for sid in subchunks:
        src = sub_src_of[sid]
        have_time[(src, sid)] = _Dec(0)
        elig_time[(src, sid)] = _Dec(0)
        # 源节点按节点类型立即/稍后入队？——源是 GPU，等价于“已经完成接收”，因此 t=0 即入队
        for w in G.successors(src):
            if have_time[(w, sid)] is None:
                enqueue(src, w, sid, _Dec(0))

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

    # ---------- Reward Calculation (Refactored, with imbalance range) ----------
    eps = 1e-12
    subchunk_completion_times = list(per_sub_max.values()) if per_sub_max else []
    incomplete = any(v == float("inf") or (hasattr(v, 'is_infinite') and v.is_infinite()) or (isinstance(v, float) and math.isinf(v)) for v in per_sub_max.values())
    global worst_reward_so_far
    if incomplete:
        # Some subchunks are not delivered
        num_unfinished = sum(1 for v in per_sub_max.values() if v == float("inf") or (hasattr(v, 'is_infinite') and v.is_infinite()) or (isinstance(v, float) and math.isinf(v)))
        # Compute base_reward from finished subchunks (ignore unfinished)
        finished_times = [float(x) for x in per_sub_max.values() if not (x == float("inf") or (hasattr(x, 'is_infinite') and x.is_infinite()) or (isinstance(x, float) and math.isinf(x)))]
        if finished_times:
            tail_finish = max(finished_times)
            avg_finish = sum(finished_times) / len(finished_times)
            if len(finished_times) > 1:
                mean_finish = avg_finish
                finish_var = sum((float(x) - mean_finish) ** 2 for x in finished_times) / len(finished_times)
                finish_std = math.sqrt(finish_var)
            else:
                finish_std = 0.0
            # send durations for finished subchunks
            from collections import defaultdict
            subchunk_send_durations = defaultdict(float)
            for log_entry in send_logs:
                t_start, t_end, t_recv_start, t_recv_end, u, v, sid = log_entry
                if sid in per_sub_max and not (per_sub_max[sid] == float("inf") or (hasattr(per_sub_max[sid], 'is_infinite') and per_sub_max[sid].is_infinite()) or (isinstance(per_sub_max[sid], float) and math.isinf(per_sub_max[sid]))):
                    duration = float(t_end - t_start)
                    subchunk_send_durations[sid] += duration
            send_durations_list = list(subchunk_send_durations.values())
            send_avg = sum(send_durations_list) / len(send_durations_list) if send_durations_list else 0.0
            send_max = max(send_durations_list) if send_durations_list else 0.0
            if len(send_durations_list) > 1:
                mean = send_avg
                send_var = sum((x - mean) ** 2 for x in send_durations_list) / len(send_durations_list)
                send_std = math.sqrt(send_var)
            else:
                send_std = 0.0
            makespan_val = float(tail_finish)
            # Imbalance metrics
            min_finish = min(finished_times)
            range_finish = tail_finish - min_finish
            range_val = range_finish
            mean_finish = avg_finish
            cv_finish = finish_std / (mean_finish + eps)
            sorted_finish = sorted(finished_times)
            n = len(sorted_finish)
            p10_idx = int(n * 0.1)
            p90_idx = int(n * 0.9)
            p10 = sorted_finish[p10_idx] if n > 1 else sorted_finish[0]
            p90 = sorted_finish[p90_idx] if n > 1 else sorted_finish[-1]
            p90_gap = (p90 - p10)
            k = max(1, int(n * 0.1))
            topk = sorted_finish[-k:]
            botk = sorted_finish[:k]
            topk_gap = (sum(topk)/len(topk) - sum(botk)/len(botk))
        else:
            makespan_val = 0.0
            range_val = 0.0
            cv_finish = 0.0
            p90_gap = 0.0
            topk_gap = 0.0
            send_avg = 0.0
            send_max = 0.0
            send_std = 0.0
        # --- Convert all time-related values to microseconds before reward calculation ---
        scale = 1e6
        makespan_val = makespan_val * scale
        range_val = range_val * scale
        p90_gap = p90_gap * scale
        topk_gap = topk_gap * scale
        send_avg = send_avg * scale
        send_max = send_max * scale
        send_std = send_std * scale
        # Weighted reward (base)
        w_makespan = 0.6
        w_imbalance = 0.3
        w_send = 0.1
        # 将不均衡项转成“时间”等价量后再归一化到 makespan
        imbalance_time = (0.4 * cv_finish * makespan_val) + (0.25 * p90_gap) + (0.2 * topk_gap) + (0.15 * range_val)
        imbalance_norm = imbalance_time / (makespan_val + eps)
        # 发送代价同样按 makespan 归一化
        send_metric = 0.5 * send_avg + 0.3 * send_max + 0.2 * send_std
        send_metric_norm = send_metric / (makespan_val + eps)
        base_reward = -(
            w_makespan * makespan_val
            + w_imbalance * imbalance_norm
            + w_send * send_metric_norm
        )
        # Penalty for unfinished subchunks
        penalty = num_unfinished * 2 * abs(worst_reward_so_far if worst_reward_so_far != 0.0 else base_reward if base_reward != 0.0 else 1.0)
        reward = base_reward - penalty
        # Do not update worst_reward_so_far for incomplete cases
        return reward
    else:
        # All subchunks delivered: normal reward calculation
        # Make sure tail_finish is a float, not Decimal
        tail_finish = float(max(subchunk_completion_times)) if subchunk_completion_times else 0.0
        avg_finish = sum(float(x) for x in subchunk_completion_times) / len(subchunk_completion_times) if subchunk_completion_times else 0.0
        if len(subchunk_completion_times) > 1:
            mean_finish = avg_finish
            finish_var = sum((float(x) - mean_finish) ** 2 for x in subchunk_completion_times) / len(subchunk_completion_times)
            finish_std = math.sqrt(finish_var)
        else:
            finish_std = 0.0
        # 2. Subchunk send durations: for each subchunk, sum all its send durations (across all links)
        from collections import defaultdict
        subchunk_send_durations = defaultdict(float)
        for log_entry in send_logs:
            t_start, t_end, t_recv_start, t_recv_end, u, v, sid = log_entry
            duration = float(t_end - t_start)
            subchunk_send_durations[sid] += duration
        send_durations_list = list(subchunk_send_durations.values())
        send_avg = sum(send_durations_list) / len(send_durations_list) if send_durations_list else 0.0
        send_max = max(send_durations_list) if send_durations_list else 0.0
        if len(send_durations_list) > 1:
            mean = send_avg
            send_var = sum((x - mean) ** 2 for x in send_durations_list) / len(send_durations_list)
            send_std = math.sqrt(send_var)
        else:
            send_std = 0.0
        makespan_val = float(makespan)
        # 4. Imbalance metrics (updated: no normalization by makespan for p90_gap, topk_gap, range)
        if subchunk_completion_times:
            sorted_finish = sorted(float(x) for x in subchunk_completion_times)
            min_finish = sorted_finish[0] if sorted_finish else 0.0
            range_finish = tail_finish - min_finish
            range_val = range_finish
            mean_finish = avg_finish
            cv_finish = finish_std / (mean_finish + eps)
            n = len(sorted_finish)
            p10_idx = int(n * 0.1)
            p90_idx = int(n * 0.9)
            p10 = sorted_finish[p10_idx] if n > 1 else sorted_finish[0]
            p90 = sorted_finish[p90_idx] if n > 1 else sorted_finish[-1]
            p90_gap = (p90 - p10)
            k = max(1, int(n * 0.1))
            topk = sorted_finish[-k:]
            botk = sorted_finish[:k]
            topk_gap = (sum(topk)/len(topk) - sum(botk)/len(botk))
        else:
            range_val = 0.0
            cv_finish = 0.0
            p90_gap = 0.0
            topk_gap = 0.0
        # --- Convert all time-related values to microseconds before reward calculation ---
        scale = 1e6
        makespan_val = makespan_val * scale
        range_val = range_val * scale
        p90_gap = p90_gap * scale
        topk_gap = topk_gap * scale
        send_avg = send_avg * scale
        send_max = send_max * scale
        send_std = send_std * scale
        # 5. Weighted reward: makespan, imbalance, send durations
        w_makespan = 0.6
        w_imbalance = 0.3
        w_send = 0.1
        # 将不均衡项转成“时间”等价量后再归一化到 makespan
        imbalance_time = (0.4 * cv_finish * makespan_val) + (0.25 * p90_gap) + (0.2 * topk_gap) + (0.15 * range_val)
        imbalance_norm = imbalance_time / (makespan_val + eps)
        # 发送代价同样按 makespan 归一化
        send_metric = 0.5 * send_avg + 0.3 * send_max + 0.2 * send_std
        send_metric_norm = send_metric / (makespan_val + eps)
        reward = -(
            w_makespan * makespan_val
            + w_imbalance * imbalance_norm
            + w_send * send_metric_norm
        )
        # Update worst_reward_so_far if this reward is more negative
        if reward < worst_reward_so_far:
            worst_reward_so_far = reward
        return reward