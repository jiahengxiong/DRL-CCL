# -*- coding: utf-8 -*-
# File: DRL-tree/main_bfs_sim.py

# ---- simple logger switch ----
VERBOSE = True

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

from decimal import Decimal, getcontext
getcontext().prec = 50  # 高精度，避免浮点误差


import heapq
import itertools
from collections import defaultdict, deque
from typing import Optional
import random

# ---- Hot-start policy (optional) ----
HOT_POLICY = None  # list of [chunk_id, src, dst, time]

def set_hot_policy(policy_list):
    """Set optional hot-start policy: list of [chunk_id, src, dst, Decimal_time].
    This is used for (1) seeding trees and (2) per-link queue tie-breaking.
    """
    global HOT_POLICY
    HOT_POLICY = policy_list or None

# ========= 辅助函数：从 policy 热启动构造初始树 =========
def build_trees_from_policy(G, all_subs, sub_src_of, packet_size_per_subchunk, gpu_nodes, policy):
    """
    从外部 policy 为每个 subchunk 构建热启动树。
    policy 的每条记录是 [chunk_id, src, dst, latency]，其中 chunk_id 从 0 开始，
    对应真实源 GPU 为 gpu_nodes[chunk_id]（因为 0~3 是 BR，不是 GPU）。

    热启动策略：
      1) 先从 policy 中抽取每个源的 (src,dst) 候选边，同时提取 BR→BR（跨 DC）候选边用于推断 BR 对；
      2) 基于推断/枚举到的 BR 对，先用 `build_tree_with_br_pair` 构一棵覆盖所有 GPU 的**合法**树；
      3) 在不违反约束的前提下，尽可能把 policy 中的 (u→v) 注入到该树里（换父重连）；
      4) 如果某源无法由任意 BR 对得到合法树，则回退到默认 BFS；

    返回: trees, depths, max_depths, choice_per_sid（每个 subchunk 选用的 BR 对，若回退则为 None）。
    """
    from collections import defaultdict, deque
    from decimal import Decimal

    # --- 解析 policy: 按真实源 GPU 分组，同时记录候选 BR→BR 跨 DC 边（带时间） ---
    per_root_edges = defaultdict(list)           # root_gpu -> list[(u,v)]
    per_root_cross = defaultdict(list)           # root_gpu -> list[(u,v, t)] 仅 BR→BR 且跨 DC

    for item in (policy or []):
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        cid, u, v = item[0], item[1], item[2]
        t = item[3] if len(item) > 3 else None
        # 将 chunk_id -> 真实源 GPU（gpu_nodes[chunk_id]）
        if not (isinstance(cid, int) and 0 <= cid < len(gpu_nodes)):
            continue
        root_gpu = gpu_nodes[cid]
        if not G.has_edge(u, v):
            continue
        per_root_edges[root_gpu].append((u, v))
        if is_border_router(G, u) and is_border_router(G, v) and is_cross_dc_edge(G, u, v):
            per_root_cross[root_gpu].append((u, v, t))

    # --- 可用的 BR 对 ---
    (dc_left, dc_right), dc_to_brs, all_pairs = enumerate_br_pairs(G)

    def _infer_br_pair_for_root(root):
        """优先用 policy 中时间最早的 BR→BR 作为 BR 对；若无则返回 None（后续枚举）。"""
        crosses = per_root_cross.get(root, [])
        if not crosses:
            return None
        # 选时间最早（t 小）的跨 DC 对；无时间则放末尾
        def _key(x):
            _, _, tt = x
            try:
                return Decimal(tt) if tt is not None else Decimal('Infinity')
            except Exception:
                return Decimal('Infinity')
        u, v, _ = sorted(crosses, key=_key)[0]
        dcu = G.nodes[u].get('DC'); dcv = G.nodes[v].get('DC')
        if dcu is None or dcv is None:
            return None
        # 规范化为 (左 DC 的 BR, 右 DC 的 BR)
        if dcu == dc_left and dcv == dc_right:
            cand = (u, v)
        elif dcv == dc_left and dcu == dc_right:
            cand = (v, u)
        else:
            return None
        # 确保该组合在 all_pairs 内（至少一个方向存在）
        for (bl, br) in all_pairs:
            if {bl, br} == {cand[0], cand[1]}:
                return (bl, br)
        return None

    def _inject_policy_edges(root, children, depth, br_pair, edges):
        """将 policy 中的 (u→v) 尽量注入当前树：把 v 的父亲改为 u，保持合法性。"""
        changed = False
        parent = build_parent_from_children(children)
        # 尝试多轮注入（最多 2 轮），避免因顺序导致的一次性失败
        for _round in range(2):
            applied = False
            for (u, v) in edges:
                if u == v:
                    continue
                if v not in depth:  # v 不在树上（不应该出现，覆盖树应含所有 GPU/BR 节点中的 GPU）
                    continue
                # 节点必须存在于拓扑，且满足 BR 对约束
                if not (G.has_edge(u, v) and edge_allowed_with_pair(G, u, v, br_pair)):
                    continue
                # 禁止产生环：u 不能在 v 的子树中
                sub_v = collect_subtree_nodes(children, v)
                if u in sub_v:
                    continue
                old_p = parent.get(v)
                if old_p == u:
                    continue
                # 备份并尝试换父
                cand = {k: list(lst) for k, lst in children.items()}
                if old_p is not None and v in cand.get(old_p, []):
                    cand[old_p] = [x for x in cand[old_p] if x != v]
                cand.setdefault(u, []).append(v)
                ok, clean, dnew, md = validate_tree_constraints(G, cand, root, br_pair, gpu_nodes)
                if ok:
                    children, depth = clean, dnew
                    parent = build_parent_from_children(children)
                    changed = applied = True
            if not applied:
                break
        return children, depth, changed

    trees, depths, max_depths, choice = {}, {}, {}, {}

    def _fallback_with_any_pair(src):
        for br_l, br_r in all_pairs:
            res = build_tree_with_br_pair(G, src, br_l, br_r, packet_size_per_subchunk, gpu_nodes)
            if res is not None:
                ch, dmap, md = res
                return ch, dmap, md, (br_l, br_r)
        # 最后退路：简单 BFS（不强制 BR 对）
        fb_t, fb_d, fb_m = build_bfs_trees(G, [sid], sub_src_of)
        return fb_t[sid], fb_d[sid], fb_m[sid], None

    # --- 对每个 subchunk 独立构树，并尽可能注入 policy ---
    for sid in all_subs:
        src = sub_src_of[sid]
        # 1) 选择 BR 对：优先 policy 推断，否则再枚举择优
        br_pair = _infer_br_pair_for_root(src)
        base = None
        best_metric = Decimal('Infinity')
        if br_pair is not None:
            base = build_tree_with_br_pair(G, src, br_pair[0], br_pair[1], packet_size_per_subchunk, gpu_nodes)
        if base is None:
            # 在所有 BR 对里选择一棵 metric 较好的
            for (bl, br) in all_pairs:
                res = build_tree_with_br_pair(G, src, bl, br, packet_size_per_subchunk, gpu_nodes)
                if res is None:
                    continue
                ch, dmap, md = res
                used_edges = [(u, v) for u, lst in ch.items() for v in lst]
                if not used_edges:
                    continue
                total = sum((edge_cost(G, u, v, packet_size_per_subchunk) for (u, v) in used_edges), Decimal(0))
                metric = (total / Decimal(len(used_edges))) * Decimal(md)
                if metric < best_metric:
                    best_metric = metric
                    base = (ch, dmap, md)
                    br_pair = (bl, br)
        if base is None:
            # 实在不行，回退
            ch, dmap, md, br_pair = _fallback_with_any_pair(src)
        else:
            ch, dmap, md = base

        # 2) 尽量注入 policy 指定的 (u→v)
        edges = per_root_edges.get(src, [])
        if edges:
            ch, dmap, _ = _inject_policy_edges(src, ch, dmap, br_pair, edges)
            # 重建 max_depth
            _, _, md = rebuild_children_and_depth(ch, src)

        # 3) 记录结果
        trees[sid] = {u: list(vs) for u, vs in ch.items()}
        depths[sid] = dict(dmap)
        max_depths[sid] = md
        choice[sid] = br_pair

    return trees, depths, max_depths, choice
def beam_search_tree_combo(G, packet_size_per_subchunk, gpu_nodes,
                           subchunks, sub_src_of, init_trees, init_depths, init_max_depths,
                           init_br_choice, beam_width=6, topk=4, extra=3, beta=4.0):
    """
    Beam search over combinations of candidate trees for slow subchunks.
    1. Identify slow subchunks by simulate_allgather_pipeline_bfs.
    2. For each slow subchunk, generate candidate trees via local operators.
    3. Iteratively expand combinations, keeping top beam_width combos by makespan.
    4. Return best combination found.
    """
    from decimal import Decimal
    # Internal: Evaluate with extra bottleneck subchunk rescheduling
    def _evaluate_with_bottleneck_reschedule(trees, depths, maxd):
        ms, per_sub = evaluate_with_edge_bottleneck_tuning(
            G, packet_size_per_subchunk, gpu_nodes, trees, depths, maxd, inner_rounds=1
        )
        avg = sum(per_sub.values()) / len(per_sub) if per_sub else Decimal(0)
        bottlenecks = [sid for sid, t in per_sub.items() if t > avg * Decimal(1.2)]
        boosts = {}
        if bottlenecks:
            # 取瓶颈 subchunk 的 top-3 链路
            for sid in bottlenecks:
                waits = [(wt, u, v) for (s, u, v), wt in getattr(simulate_allgather_pipeline_bfs, "_last_queue_wait", {}).items() if s == sid]
                waits.sort(reverse=True)
                for rank, (wt, u, v) in enumerate(waits[:3]):
                    boosts[(sid, u, v)] = len(waits) - rank
        simulate_allgather_pipeline_bfs._edge_boosts = boosts
        ms2, _ = evaluate_with_edge_bottleneck_tuning(
            G, packet_size_per_subchunk, gpu_nodes, trees, depths, maxd, inner_rounds=1
        )
        return min(ms, ms2), per_sub

    # Initial evaluate
    ms_best, per_best = _evaluate_with_bottleneck_reschedule(
        init_trees, init_depths, init_max_depths
    )
    best_pack = (init_trees, init_depths, init_max_depths, init_br_choice, ms_best)

    slow = select_slow_subchunks(per_best, min_k=3, delta=0.2)
    if not slow:
        return best_pack

    # Beam archive: list of tuples (ms, trees, depths, maxd, choice)
    archive = [(ms_best, init_trees, init_depths, init_max_depths, init_br_choice)]

    for sid in slow:
        new_archive = []
        for (ms_cur, trees_cur, depths_cur, maxd_cur, choice_cur) in archive:
            # generate candidates for this sid
            br_pair = choice_cur.get(sid, None)
            cand_set = _propose_candidates_for_tree(
                G, sub_src_of[sid], trees_cur[sid], depths_cur[sid],
                br_pair, packet_size_per_subchunk, gpu_nodes,
                per_op_cap=8
            )
            chosen = select_candidates_scored(cand_set, K=topk, extra=extra, beta=beta)
            if not chosen:
                new_archive.append((ms_cur, trees_cur, depths_cur, maxd_cur, choice_cur))
                continue

            for (c_children, c_depth, c_maxd, _score, op_name) in chosen:
                cand_trees = {k: {u: list(v) for u, v in val.items()} for k, val in trees_cur.items()}
                cand_depths = {k: dict(v) for k, v in depths_cur.items()}
                cand_maxd   = dict(maxd_cur)
                cand_choice = dict(choice_cur)

                cand_trees[sid] = c_children
                cand_depths[sid] = c_depth
                cand_maxd[sid]   = c_maxd

                ms_cand, _ = _evaluate_with_bottleneck_reschedule(
                    cand_trees, cand_depths, cand_maxd
                )
                new_archive.append((ms_cand, cand_trees, cand_depths, cand_maxd, cand_choice))

        # keep top beam_width
        new_archive.sort(key=lambda x: x[0])
        archive = new_archive[:beam_width]

    # return best from archive
    best = min(archive, key=lambda x: x[0])
    return best[1], best[2], best[3], best[4], best[0]

def optimize_tree_combo_learned(
    G, packet_size_per_subchunk: Decimal, gpu_nodes,
    subchunks, sub_src_of,
    init_trees, init_depths, init_max_depths, init_br_choice,
    max_iters: int = 25, elite_size: int = 6, alpha: float = 0.6,
    local_rewire_iters: int = 80
):
    """Cross-Entropy / policy-learning style optimizer over BR pairs per subchunk.
    For each subchunk we keep a categorical distribution over candidate BR-pairs (including a None fallback),
    sample a full combination, build trees, optionally do a few local rewire steps, evaluate makespan,
    and update the per-subchunk distributions towards the elite set.
    Returns: best_trees, best_depths, best_maxd, best_choice, best_ms
    """
    # Prepare candidate BR pairs (plus None fallback)
    (_, _), _dc_to_brs, pairs = enumerate_br_pairs(G)
    candidates = list(pairs)
    candidates_with_none = candidates + [None]

    # Initialize per-subchunk probabilities uniformly
    prob = {sid: [1.0/len(candidates_with_none)] * len(candidates_with_none) for sid in subchunks}

    # Evaluate initial combo (from init_*)
    best_trees = {k: {u: list(v) for u, v in val.items()} for k, val in init_trees.items()}
    best_depths = {k: dict(v) for k, v in init_depths.items()}
    best_maxd = dict(init_max_depths)
    best_choice = dict(init_br_choice)
    ms_best, _ = simulate_allgather_pipeline_bfs(
        G, packet_size_per_subchunk, gpu_nodes,
        trees_override=best_trees, depths_override=best_depths, max_depths_override=best_maxd,
        verbose=False
    )

    # Keep an archive of trials: (makespan, trees, depths, maxd, choice)
    archive = [(ms_best, best_trees, best_depths, best_maxd, best_choice)]

    def sample_choice_for_sid(sid):
        ws = prob[sid]
        r = random.random()
        cum = 0.0
        for i, w in enumerate(ws):
            cum += w
            if r <= cum:
                return candidates_with_none[i]
        return candidates_with_none[-1]

    for it in range(max_iters):
        # ---- Sample a full combo
        choice = {}
        for sid in subchunks:
            choice[sid] = sample_choice_for_sid(sid)

        # ---- Build trees from sampled BR choices
        trees = {}
        depths = {}
        maxd = {}
        for sid in subchunks:
            src = sub_src_of[sid]
            br_pair = choice[sid]
            if br_pair is None:
                # fallback: BFS tree for this subchunk only
                fb_trees, fb_depths, fb_max = build_bfs_trees(G, [sid], sub_src_of)
                trees[sid] = fb_trees[sid]
                depths[sid] = fb_depths[sid]
                maxd[sid] = fb_max[sid]
            else:
                res = build_tree_with_br_pair(G, src, br_pair[0], br_pair[1], packet_size_per_subchunk, gpu_nodes)
                if res is None:
                    # invalid => use BFS fallback
                    fb_trees, fb_depths, fb_max = build_bfs_trees(G, [sid], sub_src_of)
                    trees[sid] = fb_trees[sid]
                    depths[sid] = fb_depths[sid]
                    maxd[sid] = fb_max[sid]
                    choice[sid] = None
                else:
                    children, depth, md = res
                    trees[sid] = children
                    depths[sid] = depth
                    maxd[sid] = md

        # ---- Optional small local rewire per-subchunk to exploit (uses our constraints)
        # We keep BR choice fixed while rewiring.
        for sid in subchunks:
            src = sub_src_of[sid]
            br_pair = choice[sid]
            tries = min(8, local_rewire_iters)
            for _ in range(tries):
                trial = try_rewire_one(
                    G, src,
                    trees[sid], depths[sid],
                    br_pair, packet_size_per_subchunk, gpu_nodes,
                    greedy=False
                )
                if trial is None:
                    continue
                children, depth, md = trial
                trees[sid], depths[sid], maxd[sid] = children, depth, md

        # ---- Evaluate sampled combo
        ms, _ = simulate_allgather_pipeline_bfs(
            G, packet_size_per_subchunk, gpu_nodes,
            trees_override=trees, depths_override=depths, max_depths_override=maxd,
            verbose=False
        )

        archive.append((ms, trees, depths, maxd, choice))
        archive.sort(key=lambda x: x[0])
        archive = archive[:max(elite_size, 1)]

        # ---- Update distribution towards elites
        # For each subchunk, compute freq over candidates_with_none in elite set
        for sid in subchunks:
            counts = {c: 0 for c in candidates_with_none}
            total = 0
            for j in range(min(elite_size, len(archive))):
                ch = archive[j][4]
                c = ch.get(sid)
                counts[c] += 1
                total += 1
            if total == 0:
                continue
            freq = [counts[c]/total for c in candidates_with_none]
            # Smooth update
            old = prob[sid]
            new = [(1.0 - alpha) * old[i] + alpha * freq[i] for i in range(len(old))]
            # re-normalize to avoid drift
            s = sum(new)
            if s <= 0:
                new = [1.0/len(new)] * len(new)
            else:
                new = [x/s for x in new]
            prob[sid] = new

        # Track best
        if ms < ms_best:
            ms_best = ms
            best_trees = {k: {u: list(v) for u, v in val.items()} for k, val in archive[0][1].items()}
            best_depths = {k: dict(v) for k, v in archive[0][2].items()}
            best_maxd = dict(archive[0][3])
            best_choice = dict(archive[0][4])

    return best_trees, best_depths, best_maxd, best_choice, ms_best

# ===== 依赖你工程内的拓扑与配置 =====
# 工程结构：DRL-tree/topology/NVD2_1_topology.py
from topology.NVD2_1_topology import NVD2_1_topology
import config

# ---- Fast-mode knobs (to cut runtime) ----
FAST_MODE = True         # set False to run full search
FAST_K = 32               # per-link scheduler: only peek top-K candidates each wake
FAST_MAX_INNER_ROUNDS = 5
FAST_EARLY_PATIENCE = 5
FAST_MAX_ROUNDS = 8      # optimizer outer rounds
FAST_LOCAL_REWIRE = 200   # iterations per subchunk


# ========= 事件队列基建（稳定、不会再比较 dict） =========
eid = itertools.count()  # 全局单调递增事件序号

def schedule(q, when: Decimal, kind: str, **data):
    """将事件压入优先队列。四元组：(time, seq, kind, payload)"""
    heapq.heappush(q, (when, next(eid), kind, data))


# ---- helper: 判断是否为跨DC的 border-router 边（用于建树时单次使用约束）
def is_cross_dc_edge(G, u, v):
    du = G.nodes[u].get('DC')
    dv = G.nodes[v].get('DC')
    # 仅当两端都有 DC 且不相等时视为跨DC
    if du is None or dv is None or du == dv:
        return False
    # 只把非GPU设备之间的跨DC当作“border router”边（GPU 不应跨DC直连）
    tu = G.nodes[u].get('type')
    tv = G.nodes[v].get('type')
    if tu == 'GPU' or tv == 'GPU':
        return False
    return True

# ---- helper: 判断某节点是否为 border router（是否接有跨DC链路）
def is_border_router(G, n):
    # 如果该节点的任一入边或出边是跨DC（且两端都不是GPU），则认为是 border router
    for nbr in G.predecessors(n):
        if is_cross_dc_edge(G, nbr, n):
            return True
    for nbr in G.successors(n):
        if is_cross_dc_edge(G, n, nbr):
            return True
    return False

# ---- 兼容地读取链路 capacity 字段
def _get_link_capacity(G, u, v):
    e = G.edges[(u, v)]
    if 'link_capcapacity' in e:
        return e['link_capcapacity']
    if 'link_capacity' in e:
        return e['link_capacity']
    if 'capacity' in e:
        return e['capacity']
    raise KeyError(f"Edge ({u},{v}) missing capacity attribute")

# ========= 基于拓扑权重的构树优化（每个 subchunk 一棵、并在两侧 DC 仅选一个 BR，跨DC边仅一条） =========

def edge_cost(G, u, v, packet_size_per_subchunk: Decimal) -> Decimal:
    cap = _get_link_capacity(G, u, v)
    prop = G.edges[(u, v)]['propagation_latency']
    return packet_size_per_subchunk / cap + prop


def build_tree_with_br_pair(G, src, br_left, br_right, packet_size_per_subchunk: Decimal, gpu_nodes):
    """在固定 BR 选择 (br_left in DC0, br_right in DC1) 的情况下，构造一棵以 src 为根的有向最短路树。
    约束：
      - 仅允许使用一条跨DC边 (br_left -> br_right) 或 (br_right -> br_left)；
      - 其他跨DC边全部禁用；
      - 每个 DC 只用这一个 BR（即左 DC 只能使用 br_left，右 DC 只能使用 br_right）。
    返回 children, depth, max_depth。若不可达则返回 None。
    """
    # Determine DC ids of the selected BRs
    dc_l = G.nodes[br_left].get('DC')
    dc_r = G.nodes[br_right].get('DC')
    # 允许所有合法的 BR–BR 跨DC边（不限制方向）
    allowed_cross = []
    if G.has_edge(br_left, br_right) and is_cross_dc_edge(G, br_left, br_right):
        allowed_cross.append((br_left, br_right))
    if G.has_edge(br_right, br_left) and is_cross_dc_edge(G, br_right, br_left):
        allowed_cross.append((br_right, br_left))
    # 若没有满足条件的跨DC方向，则该 BR 组合对该源不可用
    if not allowed_cross:
        return None

    def edge_allowed(u, v):
        # Do not restrict which BRs can appear within a DC; only constrain cross-DC edges below
        def br_ok(_n):
            return True
        # (kept for structure; no per-DC BR limitation)
        if not br_ok(u) or not br_ok(v):
            return False
        if is_cross_dc_edge(G, u, v):
            return (u, v) in allowed_cross
        return True

    # Dijkstra（带“是否已跨过 DC”的状态），以 src 为源，得到到所有节点的最短路径父亲
    INF = Decimal('Infinity')
    # 状态：(dist, node, used_cross:boolean)
    best = defaultdict(lambda: {False: INF, True: INF})
    parent = {}
    best[src][False] = Decimal(0)
    pq = [(Decimal(0), next(eid), src, False)]

    while pq:
        d, _, u, used_cross = heapq.heappop(pq)
        if d != best[u][used_cross]:
            continue
        for v in G.successors(u):
            if not edge_allowed(u, v):
                continue
            c = edge_cost(G, u, v, packet_size_per_subchunk)
            nu = used_cross or is_cross_dc_edge(G, u, v)
            nd = d + c
            if nd < best[v][nu]:
                best[v][nu] = nd
                parent[(v, nu)] = (u, used_cross)
                heapq.heappush(pq, (nd, next(eid), v, nu))

    # 从 best 中挑选最终父亲：优先使用已跨状态（保证全树仅通过唯一跨边）
    final_parent = {}
    for n in G.nodes:
        d0, d1 = best[n][False], best[n][True]
        if n == src:
            final_parent[n] = None
            continue
        if d1 < d0 and (n, True) in parent:
            p, _ = parent[(n, True)]
            final_parent[n] = p
        elif d0 < INF and (n, False) in parent:
            p, _ = parent[(n, False)]
            final_parent[n] = p
        # 否则不可达，跳过

    # 生成 children / depth，仅对“从 src 可达的节点”
    children = defaultdict(list)
    depth = {src: 0}
    for n, p in list(final_parent.items()):
        if n == src or p is None:
            continue
        children[p].append(n)

    # BFS 计算深度
    q = deque([src])
    while q:
        u = q.popleft()
        for v in children[u]:
            depth[v] = depth[u] + 1
            q.append(v)

    # 只保留 GPU 节点在可达集合中
    reach_gpu = [g for g in gpu_nodes if g in depth]
    if len(reach_gpu) != len(gpu_nodes):
        # 有 GPU 不可达，则该组合无效
        return None

    max_depth = max(depth.values()) if depth else 0
    return children, depth, max_depth


# ---- helper: 枚举 DC->BR lists 以及所有允许的 BR 对组合 ----
def enumerate_br_pairs(G):
    dc_to_brs = defaultdict(list)
    for n in G.nodes:
        if is_border_router(G, n):
            dc = G.nodes[n].get('DC')
            if dc is not None:
                dc_to_brs[dc].append(n)
    dcs = sorted(dc_to_brs.keys())
    if len(dcs) < 2:
        return dcs, dc_to_brs, []
    left_dc, right_dc = dcs[0], dcs[1]
    pairs = []
    for br_l in dc_to_brs[left_dc]:
        for br_r in dc_to_brs[right_dc]:
            # 至少存在一个方向的跨DC边
            if (G.has_edge(br_l, br_r) and is_cross_dc_edge(G, br_l, br_r)) or \
               (G.has_edge(br_r, br_l) and is_cross_dc_edge(G, br_r, br_l)):
                pairs.append((br_l, br_r))
    return (left_dc, right_dc), dc_to_brs, pairs


# ========= Local-search helpers (模块级，构树与优化器公用) =========
def build_parent_from_children(children):
    parent = {}
    for u, lst in children.items():
        for v in lst:
            parent[v] = u
    return parent  # root = not in keys

def rebuild_children_and_depth(children, root):
    from collections import defaultdict, deque
    depth = {root: 0}
    clean = defaultdict(list)
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            if v in depth:
                continue
            depth[v] = depth[u] + 1
            clean[u].append(v)
            q.append(v)
    max_d = max(depth.values()) if depth else 0
    return clean, depth, max_d

def collect_subtree_nodes(children, node):
    res = set()
    stack = [node]
    while stack:
        u = stack.pop()
        if u in res:
            continue
        res.add(u)
        for v in children.get(u, []):
            stack.append(v)
    return res

def count_cross_edges_in_tree(G, children):
    cnt = 0
    used = []
    for u, lst in children.items():
        for v in lst:
            if is_cross_dc_edge(G, u, v):
                cnt += 1
                used.append((u, v))
    return cnt, used

def allowed_cross_by_pair(G, br_pair):
    if br_pair is None:
        return set()
    br_l, br_r = br_pair
    allowed = set()
    if G.has_edge(br_l, br_r) and is_cross_dc_edge(G, br_l, br_r):
        allowed.add((br_l, br_r))
    if G.has_edge(br_r, br_l) and is_cross_dc_edge(G, br_r, br_l):
        allowed.add((br_r, br_l))
    return allowed

def edge_allowed_with_pair(G, u, v, br_pair):
    # 必须是拓扑中存在的边
    if not G.has_edge(u, v):
        return False
    # 跨DC边必须由 BR→BR 承担；同 DC 内部不受限
    if is_cross_dc_edge(G, u, v):
        return is_border_router(G, u) and is_border_router(G, v)
    return True

def validate_tree_constraints(G, children, src, br_pair, gpu_nodes):
    clean, depth, _ = rebuild_children_and_depth(children, src)
    # 所有 GPU 可达
    for g in gpu_nodes:
        if g not in depth:
            return False, None, None, None
    # 跨DC边：仅约束必须是 BR→BR（可多条）
    cnt, used = count_cross_edges_in_tree(G, clean)
    for (u, v) in used:
        if not (is_border_router(G, u) and is_border_router(G, v)):
            return False, None, None, None
    # 所有边存在且允许
    for a, lst in clean.items():
        for b in lst:
            if not G.has_edge(a, b):
                return False, None, None, None
            if not edge_allowed_with_pair(G, a, b, br_pair):
                return False, None, None, None

    # （放宽）允许每个 DC 中出现多个 BR，BR 可为叶子，方向一致性校验移除

    max_d = max(depth.values()) if depth else 0
    return True, clean, depth, max_d

# ========= Candidate selection helpers (Top-K + soft sampling) =========
def _canonical_edge_tuple(children):
    """Return a canonical, hashable edge set for de-dup of child maps."""
    edges = []
    for u, lst in children.items():
        for v in lst:
            edges.append((int(u), int(v)))
    edges.sort()
    return tuple(edges)

def _tree_local_metric(G, child_map, src, packet_size_per_subchunk):
    """
    Cheap local proxy for a tree: average edge cost * max_depth.
    Used only for candidate ranking/selection (NOT as final objective).
    """
    used_edges = [(u, v) for u, lst in child_map.items() for v in lst]
    if not used_edges:
        return Decimal('Infinity')
    total = Decimal(0)
    for (u, v) in used_edges:
        if not G.has_edge(u, v):
            return Decimal('Infinity')
        total += edge_cost(G, u, v, packet_size_per_subchunk)
    _, _dtmp, maxd = rebuild_children_and_depth(child_map, src)
    return (total / Decimal(len(used_edges))) * Decimal(maxd)

def select_candidates_scored(scored_candidates, K=5, extra=3, beta=5.0):
    """
    Input: scored_candidates = [(children, depth, maxd, score), ...], score 越小越好
    Return: a list of the chosen subset with the same tuple format.
    """
    import random, math
    if not scored_candidates:
        return []
    # sort ascending by score
    scored_candidates = sorted(scored_candidates, key=lambda x: x[3])
    chosen = scored_candidates[:max(0, int(K))]
    rest = scored_candidates[len(chosen):]
    if not rest or extra <= 0:
        return chosen
    # softmax over negative scores (smaller score -> higher prob)
    s0 = rest[0][3]
    xs = []
    for _c in rest:
        try:
            xs.append(float(_c[3] - s0))
        except Exception:
            xs.append(0.0)
    # numerical stability
    ws = [math.exp(-float(beta) * x) for x in xs]
    s = sum(ws)
    if s <= 0:
        return chosen
    ps = [w / s for w in ws]
    idxs = list(range(len(rest)))
    # sample without replacement up to `extra`
    for _ in range(min(int(extra), len(rest))):
        r = random.random()
        cum = 0.0
        pick = None
        for i, p in zip(idxs, ps):
            cum += p
            if r <= cum:
                pick = i
                break
        if pick is None:
            pick = idxs[-1]
        chosen.append(rest[pick])
        # remove picked
        j = idxs.index(pick)
        idxs.pop(j); ps.pop(j)
        if ps:
            s = sum(ps)
            ps = [x/s for x in ps]
    return chosen

from collections import defaultdict
import random

class ContextualBandit:
    """基于上下文的多臂老虎机 (ε-greedy)"""
    def __init__(self, ops, epsilon=0.2):
        self.ops = ops
        self.epsilon = epsilon
        self.qvalues = defaultdict(lambda: {op: 0.0 for op in ops})
        self.counts = defaultdict(lambda: {op: 0 for op in ops})

    def _discretize(self, features: dict):
        # 简单离散化: 转换成 tuple
        return (
            int(features.get("depth", 0)),
            int(features.get("subtree_size", 0) > 5),
            int(features.get("is_bottleneck", 0)),
            int(features.get("edge_usage", 0) > 3),
        )

    def select(self, features: dict):
        key = self._discretize(features)
        if random.random() < self.epsilon:
            return random.choice(self.ops)
        avg_rewards = {}
        for op in self.ops:
            c = self.counts[key][op]
            r = self.qvalues[key][op]
            avg_rewards[op] = (r / c) if c > 0 else 0.0
        return max(avg_rewards.items(), key=lambda kv: kv[1])[0]

    def update(self, features: dict, op: str, reward: float):
        key = self._discretize(features)
        self.counts[key][op] += 1
        self.qvalues[key][op] += reward


class OperatorBandit:
    """简单的多臂老虎机 (ε-greedy) 算子选择器"""
    def __init__(self, ops, epsilon=0.2):
        self.ops = ops
        self.epsilon = epsilon
        self.counts = {op: 0 for op in ops}
        self.rewards = {op: 0.0 for op in ops}

    def select(self):
        import random
        if random.random() < self.epsilon:
            return random.choice(self.ops)
        avg_rewards = {op: (self.rewards[op] / self.counts[op] if self.counts[op] > 0 else 0.0) for op in self.ops}
        return max(avg_rewards.items(), key=lambda kv: kv[1])[0]

    def update(self, op, reward):
        self.counts[op] += 1
        self.rewards[op] += float(reward)


_global_bandit = None
def get_global_bandit():
    global _global_bandit
    if _global_bandit is None:
        _global_bandit = OperatorBandit(
            ["rewire_one", "two_opt", "insert_mid", "bypass_mid", "graft_sib", "none"]
        )
    return _global_bandit

# ContextualBandit 单例
_global_cbandit = None
def get_global_cbandit():
    global _global_cbandit
    if _global_cbandit is None:
        _global_cbandit = ContextualBandit(
            ["rewire_one", "two_opt", "insert_mid", "bypass_mid", "graft_sib", "none"]
        )
    return _global_cbandit

def _propose_candidates_for_tree(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes,
                                 per_op_cap=8):
    """
    Use multiple randomized local operators to generate a *set* of feasible tree candidates.
    Deduplicate by edge-set and attach a cheap local score for Top-K + soft sampling.
    Returns: list of (children, depth, maxd, score, op).
    """
    import random
    ops = ["rewire_one", "two_opt", "insert_mid", "bypass_mid", "graft_sib", "none"]
    uniq = set()
    out = []

    # ---- ContextualBandit: extract features for this tree position ----
    cbandit = get_global_cbandit()

    # Collect edge usage count for all edges in all trees (global, but here just for this call)
    # For a more global count, this dict should be managed at a higher scope (or as a global variable)
    # Here, we just count for this tree as an example
    # For a global count, you can maintain a module-level variable if needed.
    # For now, let's just use the current tree's edges.
    edge_usage_count = defaultdict(int)
    for u, lst in children.items():
        for v in lst:
            edge_usage_count[(u, v)] += 1

    # Features: depth, subtree_size, is_bottleneck, edge_usage
    # depth: max(depth.values())
    # subtree_size: total number of nodes in this subtree
    # is_bottleneck: 0 (default, can extend later)
    # edge_usage: max count of any edge in this subtree
    features = {
        "depth": max(depth.values()) if depth else 0,
        "subtree_size": len(depth),
        "is_bottleneck": 0,
        "edge_usage": max(edge_usage_count.values()) if edge_usage_count else 0,
    }

    def _try_collect(op_name):
        if op_name == "none":
            # 跳过，不生成候选
            return
        tries = 0
        got = 0
        local_cap = max(1, int(per_op_cap))
        while tries < local_cap * 3 and got < local_cap:
            tries += 1
            trial = None
            if op_name == "rewire_one":
                trial = try_rewire_one(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes, greedy=False)
            elif op_name == "two_opt":
                trial = try_edge_exchange_two_opt(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes)
            elif op_name == "insert_mid":
                trial = try_insert_intermediate_node(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes)
            elif op_name == "bypass_mid":
                trial = try_bypass_intermediate_node(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes)
            elif op_name == "graft_sib":
                trial = try_graft_sibling(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes)
            if trial is None:
                continue
            ch2, d2, md2 = trial
            key = _canonical_edge_tuple(ch2)
            if key in uniq:
                continue
            uniq.add(key)
            score = _tree_local_metric(G, ch2, src, packet_size_per_subchunk)
            out.append((ch2, d2, md2, score, op_name))
            got += 1

    # Use contextual bandit to select operator for each proposal
    for _ in range(len(ops)):
        op = cbandit.select(features)
        if op == "none":
            continue  # 跳过，不对这棵树做算子修改
        _try_collect(op)
    return out

def try_rewire_one(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes, greedy=True):
    """单步换父本地搜索（触发边必须是 GPU–BR / BR–BR / BR–GPU）。
    - greedy=True  : 保持原逻辑（用 tree_metric 过滤，仅返回更优的候选）。
    - greedy=False : 生成第一个合法换父候选（不做局部 metric 过滤，交给外层用 makespan 打分）。
    成功返回 (children, depth, maxd)，否则 None。
    """
    import random

    parent = build_parent_from_children(children)
    nodes = [n for n in depth.keys() if n != src]
    if not nodes:
        print(f"[DEBUG-CAND][REWIRE] no feasible candidate found for src={src}")
        return None
    random.shuffle(nodes)

    def tree_metric(child_map):
        used_edges = [(u, v) for u, lst in child_map.items() for v in lst]
        if not used_edges:
            return Decimal('Infinity')
        total = Decimal(0)
        for (u, v) in used_edges:
            total += edge_cost(G, u, v, packet_size_per_subchunk)
        _, _dtmp, maxd = rebuild_children_and_depth(child_map, src)
        return total / Decimal(len(used_edges)) * Decimal(maxd)

    if greedy:
        base_metric = tree_metric(children)

    for x in nodes:
        old_p = parent.get(x)
        if old_p is None:
            continue

        # ✅ 触发条件：只处理瓶颈三类边
        if not is_trigger_edge(old_p, x, G):
            continue

        sub_nodes = collect_subtree_nodes(children, x)

        preds = list(G.predecessors(x))
        random.shuffle(preds)
        for p2 in preds:
            if p2 == old_p or p2 in sub_nodes:
                continue
            if not G.has_edge(p2, x):
                continue
            if not edge_allowed_with_pair(G, p2, x, br_pair):
                continue

            cand = {k: list(v) for k, v in children.items()}
            if x in cand.get(old_p, []):
                cand[old_p] = [z for z in cand[old_p] if z != x]
            cand.setdefault(p2, []).append(x)

            ok, clean, dnew, maxd = validate_tree_constraints(G, cand, src, br_pair, gpu_nodes)
            if not ok:
                continue

            if not greedy:
                print(f"[DEBUG-CAND][REWIRE] generated candidate for node {x} with new parent {p2}")
                return (clean, dnew, maxd)

            m = tree_metric(clean)
            if m + Decimal('1e-18') < base_metric:
                print(f"[DEBUG-CAND][REWIRE] generated candidate for node {x} with new parent {p2}")
                return (clean, dnew, maxd)

    print(f"[DEBUG-CAND][REWIRE] no feasible candidate found for src={src}")
    return None


# ========= 构造所有 subchunk 的 BFS 树（每个 subchunk 一棵树；跨DC边在同一棵树内只允许一次） =========
def build_bfs_trees(G, subchunks, sub_src_of):
    """
    返回：
      trees: dict[sub_id] -> children_map(dict: u -> list(children))
      depths: dict[sub_id] -> dict(node -> depth)
      max_depths: dict[sub_id] -> int
    规则：对每个 subchunk（由其源 GPU 决定）各自建一棵 BFS 树；
    在**同一棵树**内，跨DC（border router 间）的有向边最多只允许出现一次。
    """
    adj = defaultdict(list)
    for u, v in G.edges:
        adj[u].append(v)

    trees = {}
    depths = {}
    max_depths = {}

    for sub_id in subchunks:
        src = sub_src_of[sub_id]
        parent = {src: None}
        depth = {src: 0}
        children = defaultdict(list)

        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                # 仅当为跨DC时，要求 BR→BR；否则正常 BFS
                if is_cross_dc_edge(G, u, v) and not (is_border_router(G, u) and is_border_router(G, v)):
                    continue
                if v in parent:
                    continue
                parent[v] = u
                depth[v] = depth[u] + 1
                children[u].append(v)
                q.append(v)

        trees[sub_id] = children
        depths[sub_id] = depth
        max_depths[sub_id] = max(depth.values()) if depth else 0

    return trees, depths, max_depths


def build_optimized_trees(G, subchunks, sub_src_of, packet_size_per_subchunk: Decimal, gpu_nodes):
    """为每个 subchunk 选一对 BR，并按约束构一棵更优的树；若无有效 BR 对则回退到 BFS。"""
    """
    为每个 subchunk 选一对 BR，并按约束构一棵更优的树；若无有效 BR 对则回退到 BFS。
    """
    # Hot-start from policy if available: build per-subchunk trees from provided policy edges
    if 'HOT_POLICY' in globals() and HOT_POLICY:
        trees, depths, max_depths, choice = build_trees_from_policy(
            G, subchunks, sub_src_of, packet_size_per_subchunk, gpu_nodes, HOT_POLICY
        )
        # If policy yields valid coverage, return immediately
        if trees and depths and max_depths:
            return trees, depths, max_depths, choice

    (_, _), _dc_to_brs, pairs = enumerate_br_pairs(G)
    if not pairs:
        trees, depths, max_depths = build_bfs_trees(G, subchunks, sub_src_of)
        return trees, depths, max_depths, {sid: None for sid in subchunks}

    trees, depths, max_depths = {}, {}, {}
    br_choice = {}

    for sub_id in subchunks:
        src = sub_src_of[sub_id]
        best_tree = None
        best_metric = Decimal('Infinity')
        best_depths = None
        best_maxd = None
        best_pair = None

        for br_l, br_r in pairs:
            res = build_tree_with_br_pair(G, src, br_l, br_r, packet_size_per_subchunk, gpu_nodes)
            if res is None:
                continue
            children, depth, maxd = res
            used_edges = [(u, v) for u, lst in children.items() for v in lst]
            if not used_edges:
                continue
            total = sum((edge_cost(G, u, v, packet_size_per_subchunk) for (u, v) in used_edges), Decimal(0))
            avg_edge = total / Decimal(len(used_edges))
            metric = avg_edge * Decimal(maxd)
            if metric < best_metric:
                best_metric = metric
                best_tree = children
                best_depths = depth
                best_maxd = maxd
                best_pair = (br_l, br_r)

        if best_tree is None:
            fb_trees, fb_depths, fb_maxd = build_bfs_trees(G, [sub_id], sub_src_of)
            best_tree = fb_trees[sub_id]
            best_depths = fb_depths[sub_id]
            best_maxd = fb_maxd[sub_id]
            best_pair = None

        trees[sub_id] = best_tree
        depths[sub_id] = best_depths
        max_depths[sub_id] = best_maxd
        br_choice[sub_id] = best_pair

    return trees, depths, max_depths, br_choice




#
# ===== Cross-subchunk priority (bottleneck-aware) =====
from decimal import Decimal as _Dec

def _compute_tardy_priorities(per_sub_times: dict) -> dict:
    """
    Compute per-subchunk priority based on how much slower it is than the average.
    priority = 1 + max(0, (t - avg)/avg). Returns dict[sub_id] -> Decimal priority (>=1).
    """
    if not per_sub_times:
        return {}
    vals = [t for t in per_sub_times.values() if t is not None]
    if not vals:
        return {}
    avg = sum(vals) / _Dec(len(vals))
    if avg <= 0:
        return {sid: _Dec(1) for sid in per_sub_times}
    pri = {}
    for sid, t in per_sub_times.items():
        if t is None:
            pri[sid] = _Dec(1)
            continue
        over = (t - avg) / avg
        if over < 0:
            over = _Dec(0)
        pri[sid] = _Dec(1) + over
    return pri

# ========= 模拟核心 =========
def simulate_allgather_pipeline_bfs(G, packet_size_per_subchunk: Decimal, gpu_nodes,
                                    trees_override=None, depths_override=None, max_depths_override=None,
                                    verbose=True):
    """
    G: 拓扑（DiGraph），边含:
       - 'link_capcapacity' / 'link_capacity' (Decimal): 链路带宽
       - 'propagation_latency' (Decimal)
    packet_size_per_subchunk: 每个 subchunk 的字节/大小（与 NVD2_1_topology 中一致）
    gpu_nodes: GPU 节点列表
    trees_override, depths_override, max_depths_override: 可选，若给定则直接用外部树
    verbose: 控制日志输出
    """
    global VERBOSE
    VERBOSE = bool(verbose)
    # Ensure a monotonic event id counter exists for stable heap ordering
    import itertools, heapq
    if not hasattr(simulate_allgather_pipeline_bfs, "_eid"):
        simulate_allgather_pipeline_bfs._eid = itertools.count()
    eid = simulate_allgather_pipeline_bfs._eid
    from collections import defaultdict, deque

    # === 定义所有 subchunk 及其源 ===
    num_chunks_per_gpu = 1
    num_subchunks_per_chunk = int(config.num_chunk)

    subchunks = []
    sub_src_of = {}

    for src in gpu_nodes:
        for _c in range(num_chunks_per_gpu):
            for _k in range(num_subchunks_per_chunk):
                sub_id = len(subchunks)
                subchunks.append(sub_id)
                sub_src_of[sub_id] = src

    # 生成或使用外部给定的树
    if trees_override is None:
        trees, depths, max_depths, _ = build_optimized_trees(G, subchunks, sub_src_of, packet_size_per_subchunk, gpu_nodes)
    else:
        trees, depths, max_depths = trees_override, depths_override, max_depths_override

    # Optional: build policy time map for prioritizing per-link queues
    policy_time_map = {}
    if 'HOT_POLICY' in globals() and HOT_POLICY:
        # Map GPU -> its chunk_id index (chunk_id == index in gpu_nodes)
        gpu_to_chunkid = {g: i for i, g in enumerate(gpu_nodes)}
        for sid in subchunks:
            src_gpu = sub_src_of[sid]
            chunk_id = gpu_to_chunkid.get(src_gpu, None)
            if chunk_id is None:
                continue
            for rec in HOT_POLICY:
                if not isinstance(rec, (list, tuple)) or len(rec) < 4:
                    continue
                cid, u, v, t = rec[0], rec[1], rec[2], rec[3]
                if cid == chunk_id and isinstance(u, int) and isinstance(v, int) and (u, v) in G.edges:
                    policy_time_map[(sid, u, v)] = t
    # 基于每棵树的“从该节点到叶子的最长传输时间”作为优先级（比“剩余深度”更精确）
    def _compute_downstream_time_for_sub(sub_id):
        memo = {}
        def dfs(u):
            if u in memo:
                return memo[u]
            best = Decimal(0)
            for v in trees[sub_id].get(u, []):
                c = edge_cost(G, u, v, packet_size_per_subchunk)
                cand = c + dfs(v)
                if cand > best:
                    best = cand
            memo[u] = best
            return best
        # 对该 subchunk 的所有出现在深度表的节点跑一遍
        for u in depths[sub_id].keys():
            dfs(u)
        return memo

    # 对所有 subchunk 预计算：节点 -> (到叶子的最长剩余时间)
    rem_time_map = {sid: _compute_downstream_time_for_sub(sid) for sid in subchunks}

    # 链路队列与忙闲
    link_queues = defaultdict(list)          # (u,v) -> heap [(-rem_depth, seq, payload)]
    link_free_time = defaultdict(lambda: Decimal(0))  # (u,v) -> 可用时间

    # ---- queue wait accounting (per-node and per-edge), and edge-level boosts from previous run ----
    queue_wait_by_node = defaultdict(lambda: defaultdict(Decimal))  # sid -> node -> wait_time
    queue_wait_by_edge = defaultdict(Decimal)  # (sid,u,v) -> wait_time
    edge_boosts = getattr(simulate_allgather_pipeline_bfs, "_edge_boosts", {}) or {}

    # 记录每个 (node, src_subchunk) 获取完成的时间
    have_time = defaultdict(lambda: None)  # None 表示还没有

    # 事件队列
    event_q = []

    # 计算一次链路传输 + 传播延迟
    def tx_time(u, v):
        cap = _get_link_capacity(G, u, v)
        prop = G.edges[(u, v)]['propagation_latency']
        # Return serialization time **only**; propagation handled separately.
        ser = packet_size_per_subchunk / cap
        return ser, prop

    # 尝试为某条链路调度一个任务
    def try_schedule_link(u, v, now: Decimal):
        q = link_queues[(u, v)]
        # In fast mode we skip the full stable reordering on every wake-up to save time.
        if not q:
            return

        t_free = link_free_time[(u, v)]
        t_ready = max(now, t_free)

        # Consider only top-K items to reduce sorting/partition cost
        K = FAST_K if 'FAST_K' in globals() else 8
        buf = []
        take = min(K, len(q))
        for _ in range(take):
            prio, seqno, payload = heapq.heappop(q)
            buf.append((prio, seqno, payload))
        # NOTE: unpopped items remain in the heap for later attempts

        ready = []
        waiting = []
        for prio, seqno, payload in buf:
            avail = payload['avail_time']
            if avail <= t_ready:
                ready.append((prio, seqno, payload))
            else:
                waiting.append((prio, seqno, payload))

        chosen = None
        if ready:
            # Bottleneck-aware priority among subchunks sharing the same (u,v) at this time.
            # Sort by: (tardy_priority, downstream_remaining_time, depth_delta, policy_bias), all in descending order.
            tardy_pri = getattr(simulate_allgather_pipeline_bfs, "_tardy_pri", {}) or {}

            def _prio_tuple(item):
                _prio, _seq, _payload = item
                _sid = _payload['src_sub']
                # per-sub 拖尾权重 (>=1)
                _tp = tardy_pri.get(_sid, _Dec(1))
                # 子分支剩余时间
                _rt = _payload.get('rem_time', _Dec(0))
                # 深度差（优先发向更深的分支）
                du = depths[_sid].get(u, 0) if _sid in depths else 0
                dv = depths[_sid].get(v, 0) if _sid in depths else 0
                _dd = dv - du
                _pb = _payload.get('pol_bias', _Dec(0))
                # 边级瓶颈 boost（只对 (sid,u,v) 生效）
                _eb = _Dec(edge_boosts.get((_sid, u, v), 0))
                # 按：拖尾权重、剩余时间、深度差、边级boost、policy bias（全部降序）排序
                return (_tp, _rt, _dd, _eb, _pb)
            if 'FAST_MODE' in globals() and FAST_MODE:
                # partial sort already good enough for top-1 choice
                ready.sort(key=lambda it: _prio_tuple(it), reverse=True)
            else:
                ready.sort(key=lambda it: _prio_tuple(it), reverse=True)

            chosen = ready.pop(0)
            for item in ready:
                heapq.heappush(q, item)
            for item in waiting:
                heapq.heappush(q, item)
        else:
            if waiting:
                waiting.sort(key=lambda x: (x[2]['avail_time'], x[0], x[1]))
                earliest_avail = waiting[0][2]['avail_time']
                for item in waiting:
                    heapq.heappush(q, item)
                # Do not touch the rest of heap; just wake up at earliest among inspected
                schedule(event_q, earliest_avail, "link_wakeup", u=u, v=v)
            return

        _, _, payload = chosen
        src_sub = payload['src_sub']
        avail = payload['avail_time']

        # queue-wait time on this edge for this subchunk (ready but link busy)
        # t_ready = max(now, t_free) 已在上文计算
        t_start = max(t_ready, avail)
        wait_here = t_start - max(link_free_time[(u, v)], avail)
        if wait_here > Decimal(0):
            queue_wait_by_node[src_sub][u] += wait_here
            queue_wait_by_edge[(src_sub, u, v)] += wait_here

        t_start = max(t_ready, avail)

        t_tx, t_prop = tx_time(u, v)
        # Sender finishes after serialization only
        t_end = t_start + t_tx
        # Receiver starts after propagation; finishes after serialization has fully arrived (ser + prop)
        t_recv_start = t_start + t_prop
        t_recv_end = t_start + t_tx + t_prop

        link_free_time[(u, v)] = t_end

        schedule(event_q, t_end, "send_end", u=u, v=v, src_sub=src_sub)
        schedule(event_q, t_recv_start, "recv_start", u=u, v=v, src_sub=src_sub)
        schedule(event_q, t_recv_end,   "recv_end",   u=u, v=v, src_sub=src_sub)

        if VERBOSE and not ('FAST_MODE' in globals() and FAST_MODE):
            log(
                f"node {u} 在时刻 {float(t_start)*1e6:.3f} us 发送 subchunk {src_sub} 到 node {v}，在 {float(t_end)*1e6:.3f} us 发送完，"
                f"在 {float(t_recv_start)*1e6:.3f} us 开始接收，在 {float(t_recv_end)*1e6:.3f} us 接收完"
            )
        return

    # 当某节点 node 拿到了 src_sub 的 subchunk，立即向树中的所有子节点发
    def enqueue_children_sends(node, src_sub, now: Decimal):
        from decimal import Decimal  # ensure Decimal is in scope
        children = trees[src_sub].get(node, [])
        if not children:
            return
        # 以 **子节点分支** 的剩余传播时间作为优先级（更贴合“子树深度”语义）
        for ch in children:
            # 兜底保护：确保(node, ch)是拓扑中的有效边
            if not G.has_edge(node, ch):
                continue
            rem_time_child = rem_time_map[src_sub].get(ch, Decimal(0))
            # Policy bias: earlier scheduled time in policy -> higher priority (smaller time)
            pol_t = policy_time_map.get((src_sub, node, ch), None)
            # Convert to a numeric bias: if time is provided, use negative value so earlier time sorts first
            pol_bias = -pol_t if pol_t is not None else Decimal(0)
            payload = {
                'src_sub': src_sub,
                'avail_time': now,
                'rem_time': rem_time_child,
                'pol_bias': pol_bias,
            }
            heapq.heappush(link_queues[(node, ch)],
                           (-rem_time_child, next(eid), payload))
            try_schedule_link(node, ch, now)

    # 初始化：每个 subchunk 在其源 GPU 处于时刻 0 即可用
    for sub_id in subchunks:
        src = sub_src_of[sub_id]
        have_time[(src, sub_id)] = Decimal(0)
        enqueue_children_sends(src, sub_id, Decimal(0))

    # 事件循环
    while event_q:
        t_now, _, kind, data = heapq.heappop(event_q)

        if kind == "send_end":
            u, v, src_sub = data['u'], data['v'], data['src_sub']
            try_schedule_link(u, v, t_now)

        elif kind == "recv_start":
            u, v, src_sub = data['u'], data['v'], data['src_sub']
            if is_border_router(G, v):
                enqueue_children_sends(v, src_sub, t_now)

        elif kind == "link_wakeup":
            u, v = data['u'], data['v']
            try_schedule_link(u, v, t_now)

        elif kind == "recv_end":
            u, v, src_sub = data['u'], data['v'], data['src_sub']
            key = (v, src_sub)
            if have_time[key] is None:
                have_time[key] = t_now
                if not is_border_router(G, v):
                    enqueue_children_sends(v, src_sub, t_now)

        else:
            raise ValueError(f"Unknown event kind: {kind}")

    # 统计完成时间
    per_sub_max = {}
    for s in subchunks:
        times = []
        for n in gpu_nodes:
            t = have_time[(n, s)]
            times.append(t if t is not None else Decimal('Infinity'))
        per_sub_max[s] = max(times)
    makespan = max(per_sub_max.values()) if per_sub_max else Decimal(0)

    # ---- Derive edge-level bottleneck boosts for next simulation (per tardy subchunk; Top-3 critical edges) ----
    try:
        vals = [t for t in per_sub_max.values() if t is not None]
        avg_t = sum(vals) / Decimal(len(vals)) if vals else Decimal(0)
        new_edge_boosts = {}
        for sid, tfin in per_sub_max.items():
            if tfin is None or avg_t <= 0 or tfin <= avg_t:
                continue
            # 这个 subchunk 在各边上的等待情况（由本轮收集的 queue_wait_by_edge 提供）
            waits = [((ss, u, v), w) for (ss, u, v), w in queue_wait_by_edge.items() if ss == sid]
            if not waits:
                continue
            # 等待时间降序；选取 Top-3 作为关键链路
            waits.sort(key=lambda kv: kv[1], reverse=True)
            sid_total_wait = sum(w for (ss, _, _), w in queue_wait_by_edge.items() if ss == sid) or Decimal('1e-12')
            over = (tfin - avg_t) / max(Decimal('1e-12'), avg_t)
            if over < 0:
                over = Decimal(0)
            scale = min(Decimal('2.5'), Decimal('1.0') + over)  # 拖尾越大，放大系数越大（上限 2.5）
            topk = 3
            for i in range(min(topk, len(waits))):
                (s, u, v), w_edge = waits[i]
                share = (w_edge / sid_total_wait)
                weight = scale * share
                # 累加式写入（不同 sid 在同一条边上会各自赋权）
                new_edge_boosts[(sid, u, v)] = float(weight)
        # 写回到函数属性，供下一次仿真使用
        simulate_allgather_pipeline_bfs._edge_boosts = new_edge_boosts
        # 也保存一些调试信息
        simulate_allgather_pipeline_bfs._queue_hot_edge = dict(queue_wait_by_edge)
        simulate_allgather_pipeline_bfs._per_sub_times = dict(per_sub_max)
    except Exception:
        pass

    # Persist bottleneck priorities for subsequent simulation calls
    try:
        simulate_allgather_pipeline_bfs._tardy_pri = _compute_tardy_priorities(per_sub_max)
    except Exception:
        pass

    log("\n===== Summary =====")
    for s in sorted(per_sub_max):
        log(f"subchunk {s} fully delivered by time {float(per_sub_max[s])*1e6:.3f} us")
    log(f"TOTAL makespan: {float(makespan)*1e6:.3f} us")

    # ---- Final detailed trace of send/receive times ----
    if verbose:
        try:
            for (sid, u, v), wait in queue_wait_by_edge.items():
                # recover timing from state: send/recv times were already scheduled
                send_start = None
                recv_start = None
                recv_end = None
                # approximate from have_time and queue_wait_by_node
                # fallback: use queue_wait_by_edge info
                send_start = None
                # In this simplified trace, just reuse per-sub completion times as end
                recv_end = per_sub_max.get(sid, None)
                log(f"[FinalTrace] subchunk {sid}: {u}->{v}, recv_end {float(recv_end)*1e6:.3f} us" if recv_end else f"[FinalTrace] subchunk {sid}: {u}->{v}, no completion time")
        except Exception as e:
            log(f"[TraceError] {e}")

    return makespan, per_sub_max


# ---- Queue priority helpers (bottleneck-first, stable) ----
def _link_queue_priority_key(u, v, item):
    edge_boosts = getattr(simulate_allgather_pipeline_bfs, "_edge_boosts", {}) or {}
    tardy_pri   = getattr(simulate_allgather_pipeline_bfs, "_tardy_pri", {}) or {}
    sid = None
    if isinstance(item, tuple) and item:
        sid = item[0]
    elif isinstance(item, dict):
        sid = item.get("sid", None)
    is_b = 1 if (((sid, u, v) in edge_boosts) or (sid in tardy_pri)) else 0
    w = float(edge_boosts.get((sid, u, v), 0.0)) + 0.2 * float(tardy_pri.get(sid, 0.0))
    # Negative for descending sort
    return (-is_b, -w)

def _push_link_with_priority(send_queue, u, v, item):
    """
    Insert `item` into per-link queue (u,v) with 'bottleneck-first, stable' rule:
      - All bottleneck items (identified via _edge_boosts/_tardy_pri) are kept before all non-bottlenecks.
      - Among bottlenecks, order by descending weight (edge_boost + 0.2 * tardy_pri), stable w.r.t. original order.
      - Non-bottlenecks keep original FIFO order.
    """
    q = send_queue.setdefault((u, v), [])
    edge_boosts = getattr(simulate_allgather_pipeline_bfs, "_edge_boosts", {}) or {}
    tardy_pri   = getattr(simulate_allgather_pipeline_bfs, "_tardy_pri", {}) or {}
    sid = None
    if isinstance(item, tuple) and item:
        sid = item[0]
    elif isinstance(item, dict):
        sid = item.get("sid", None)

    is_b_new = ((sid, u, v) in edge_boosts) or (sid in tardy_pri)

    if not q or not is_b_new:
        # Empty queue or non-bottleneck: preserve FIFO for non-bottlenecks
        q.append(item)
        return

    # Split queue into bottleneck head and non-bottleneck tail
    head, tail = [], []
    for it in q:
        it_sid = it[0] if (isinstance(it, tuple) and it) else (it.get("sid", None) if isinstance(it, dict) else None)
        it_is_b = ((it_sid, u, v) in edge_boosts) or (it_sid in tardy_pri)
        (head if it_is_b else tail).append(it)

    # Insert new bottleneck into head with stable, weight-descending order
    head.append(item)
    head = [x for _, x in sorted(list(enumerate(head)),
                                 key=lambda p: (_link_queue_priority_key(u, v, p[1]), p[0]))]
    send_queue[(u, v)] = head + tail

def _stable_reorder_link_queue(send_queue, u, v):
    """
    Re-stabilize the per-link queue (u,v) just before dispatch:
      - All bottlenecks first (as above), non-bottlenecks keep original relative order.
      - Stable w.r.t. original indices.
    """
    q = send_queue.get((u, v), None)
    if not q or len(q) <= 1:
        return
    q_annot = list(enumerate(q))
    q_annot.sort(key=lambda p: (_link_queue_priority_key(u, v, p[1]), p[0]))
    send_queue[(u, v)] = [item for _, item in q_annot]

# ========= Edge-level bottleneck tuning helpers (no child reordering) =========
def _reset_sim_bottleneck_state():
    """
    Clear per-run global states inside simulate_allgather_pipeline_bfs so that
    each candidate evaluation starts fresh (no cross-candidate contamination).
    """
    try:
        simulate_allgather_pipeline_bfs._edge_boosts = {}
    except Exception:
        pass
    try:
        simulate_allgather_pipeline_bfs._tardy_pri = {}
    except Exception:
        pass
    try:
        simulate_allgather_pipeline_bfs._queue_hot_edge = {}
    except Exception:
        pass
    try:
        simulate_allgather_pipeline_bfs._per_sub_times = {}
    except Exception:
        pass


def evaluate_with_edge_bottleneck_tuning(G, packet_size_per_subchunk, gpu_nodes,
                                         trees, depths, max_depths,
                                         inner_rounds: int = 3, early_stop_patience: int = 5):
    """
    Keep the tree structure fixed (do NOT reorder children).
    Run the simulator multiple times so that edge-level bottleneck boosts
    (queue_wait_by_edge -> _edge_boosts) and tardy priorities can adapt.
    Returns: (ms_best, per_best)

    NOTE:
      - This uses the existing mechanism inside simulate_allgather_pipeline_bfs:
        * queue_wait_by_edge stats -> _edge_boosts[(sid,u,v)]
        * tardy priorities (_tardy_pri)
      - No child list reordering is performed here.
    """
    from decimal import Decimal
    _reset_sim_bottleneck_state()
    ms_best, per_best = None, None
    rounds = max(1, int(inner_rounds) * len(gpu_nodes) * int(getattr(config, 'num_chunk', 1)))
    if 'FAST_MODE' in globals() and FAST_MODE:
        rounds = min(rounds, FAST_MAX_INNER_ROUNDS)
    no_improve_cnt = 0
    for r in range(rounds):
        ms, per = simulate_allgather_pipeline_bfs(
            G, packet_size_per_subchunk, gpu_nodes,
            trees_override=trees, depths_override=depths, max_depths_override=max_depths,
            verbose=False
        )
        if (ms_best is None) or (ms + Decimal('1e-18') < ms_best):
            ms_best, per_best = ms, per
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1
            patience = FAST_EARLY_PATIENCE if (('FAST_MODE' in globals() and FAST_MODE) and early_stop_patience is not None) else early_stop_patience
            if no_improve_cnt >= (patience or early_stop_patience):
                break
        # Next loop will reuse the freshly updated _edge_boosts/_tardy_pri
        # produced by the previous call.
    return ms_best, per_best


# ========= 优化器：坐标下降优化树组合 + 本地 rewiring =========
def optimize_tree_combo(G, packet_size_per_subchunk: Decimal, gpu_nodes,
                        subchunks, sub_src_of, init_trees, init_depths, init_max_depths,
                        init_br_choice, max_rounds=8, local_rewire_iters=120):
    """Coordinate-descent over BR pairs + local rewiring search per tree."""
    # current best combo
    best_trees = {k: {u: list(v) for u, v in val.items()} for k, val in init_trees.items()}
    best_depths = {k: dict(v) for k, v in init_depths.items()}
    best_maxd = dict(init_max_depths)
    best_choice = dict(init_br_choice)

    # evaluate
    ms_best, _ = evaluate_with_edge_bottleneck_tuning(
        G, packet_size_per_subchunk, gpu_nodes,
        best_trees, best_depths, best_maxd,
        inner_rounds=3
    )

    (_, _), _dc_to_brs, pairs = enumerate_br_pairs(G)

    # In fast mode, pre-rank BR pairs by a cheap static metric and keep only top-M
    if 'FAST_MODE' in globals() and FAST_MODE and pairs:
        def _pair_cost(pair):
            br_l, br_r = pair
            # approximate: sum of edge_cost from BRs to their neighbors
            tot = Decimal(0)
            for u, v in G.out_edges(br_l):
                tot += edge_cost(G, u, v, packet_size_per_subchunk)
            for u, v in G.out_edges(br_r):
                tot += edge_cost(G, u, v, packet_size_per_subchunk)
            return tot
        pairs = sorted(pairs, key=_pair_cost)[:max(2, min(4, len(pairs)))]

    improved = True
    rounds = 0
    max_outer = FAST_MAX_ROUNDS if ('FAST_MODE' in globals() and FAST_MODE) else max_rounds
    while improved and rounds < max_outer:
        improved = False
        rounds += 1

        # ---- Step 1: coordinate descent over BR pairs
        for sid in subchunks:
            cur_pair = best_choice.get(sid)
            for br_l, br_r in pairs:
                if cur_pair == (br_l, br_r):
                    continue
                src = sub_src_of[sid]
                res = build_tree_with_br_pair(G, src, br_l, br_r, packet_size_per_subchunk, gpu_nodes)
                if res is None:
                    continue
                children, depth, maxd = res

                cand_trees = {k: {u: list(v) for u, v in val.items()} for k, val in best_trees.items()}
                cand_depths = {k: dict(v) for k, v in best_depths.items()}
                cand_maxd = dict(best_maxd)
                cand_choice = dict(best_choice)

                cand_trees[sid] = children
                cand_depths[sid] = depth
                cand_maxd[sid] = maxd
                cand_choice[sid] = (br_l, br_r)

                ms_cand, _ = evaluate_with_edge_bottleneck_tuning(
                    G, packet_size_per_subchunk, gpu_nodes,
                    cand_trees, cand_depths, cand_maxd,
                    inner_rounds=2
                )
                if ms_cand < ms_best:
                    ms_best = ms_cand
                    best_trees, best_depths, best_maxd, best_choice = cand_trees, cand_depths, cand_maxd, cand_choice
                    improved = True
                    break  # accept-first

        # ---- Step 2: local rewiring per subchunk (proposal set + Top-K sampling + annealing) ----
        import random, math
        T0 = float(ms_best) * 0.10 if ms_best != 0 else 1.0  # 初始温度按当前最优 makespan 的 10%
        cooling = 0.95
        any_rewire = False
        # hyper-parameters for candidate selection
        TOPK = 4
        EXTRA = 3
        BETA  = 4.0
        per_op_cap = 6 if ('FAST_MODE' in globals() and FAST_MODE) else 12

        for sid in subchunks:
            src = sub_src_of[sid]
            br_pair = best_choice.get(sid)
            T = T0
            iters = 0
            no_progress = 0
            lr_cap = FAST_LOCAL_REWIRE if ('FAST_MODE' in globals() and FAST_MODE) else local_rewire_iters

            while iters < lr_cap and T > 1e-12 and no_progress < (lr_cap // 2):
                iters += 1

                # 1) 生成多算子候选集合（不进行全局评估）
                cand_set = _propose_candidates_for_tree(
                    G, src,
                    best_trees[sid], best_depths[sid],
                    br_pair, packet_size_per_subchunk, gpu_nodes,
                    per_op_cap=per_op_cap
                )

                if not cand_set:
                    no_progress += 1
                    T *= cooling
                    continue

                # 2) Top-K + soft sampling 选出少量候选进行 *全局* evaluate
                chosen = select_candidates_scored(cand_set, K=TOPK, extra=EXTRA, beta=BETA)

                # 3) 逐个 evaluate 并按模拟退火接受
                accepted = False
                best_local_ms = None
                best_local_pack = None

                for (c_children, c_depth, c_maxd, _score, op_name) in chosen:
                    cand_trees = {k: {u: list(v) for u, v in val.items()} for k, val in best_trees.items()}
                    cand_depths = {k: dict(v) for k, v in best_depths.items()}
                    cand_maxd   = dict(best_maxd)

                    cand_trees[sid] = c_children
                    cand_depths[sid] = c_depth
                    cand_maxd[sid]   = c_maxd

                    ms_cand, _ = evaluate_with_edge_bottleneck_tuning(
                        G, packet_size_per_subchunk, gpu_nodes,
                        cand_trees, cand_depths, cand_maxd,
                        inner_rounds=2
                    )
                    # Bandit reward update
                    reward = float(ms_best - ms_cand)
                    get_global_bandit().update(op_name, reward)

                    # 记录最好的候选（用于确定性替换），同时也允许次优以退火概率接受
                    if (best_local_ms is None) or (ms_cand < best_local_ms):
                        best_local_ms   = ms_cand
                        best_local_pack = (cand_trees, cand_depths, cand_maxd)

                    delta = float(ms_cand - ms_best)
                    accept = False
                    if ms_cand + Decimal('1e-18') < ms_best:
                        accept = True
                    else:
                        prob = math.exp(-max(0.0, delta) / max(1e-12, T))
                        if random.random() < prob:
                            accept = True

                    if accept:
                        ms_best = ms_cand
                        best_trees, best_depths, best_maxd = cand_trees, cand_depths, cand_maxd
                        any_rewire = True
                        accepted = True
                        no_progress = 0
                        break  # 本轮已接受，进入下一轮

                if not accepted:
                    # 若本轮没有以退火接受任何一个候选，但存在一个 *确定性* 最优则也接受
                    if best_local_pack is not None and best_local_ms + Decimal('1e-18') < ms_best:
                        ms_best = best_local_ms
                        best_trees, best_depths, best_maxd = best_local_pack
                        any_rewire = True
                        no_progress = 0
                    else:
                        no_progress += 1

                T *= cooling

        improved = improved or any_rewire

    return best_trees, best_depths, best_maxd, best_choice, ms_best



# ========= 更激进的大邻域搜索（LNS + Beam on critical） =========

# --- Helper: 多步换父链（SPR 链） ---
def try_rewire_chain(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes, steps=3):
    """
    多步换父（SPR 链）：连续做 steps 次非贪心 try_rewire_one，每一步都基于上一步结果。
    成功返回 (children, depth, maxd)，否则 None。
    """
    cur_children = {u: list(v) for u, v in children.items()}
    cur_depth = dict(depth)
    cur_maxd = max(depth.values()) if depth else 0

    for _ in range(max(1, int(steps))):
        trial = try_rewire_one(
            G, src,
            cur_children, cur_depth,
            br_pair, packet_size_per_subchunk, gpu_nodes,
            greedy=False
        )
        if trial is None:
            return None
        cur_children, cur_depth, cur_maxd = trial
    return cur_children, cur_depth, cur_maxd



# --- Helper: 2-opt 边交换（TBR 精简版） ---
# ---- Evaluate makespan helper ----
def evaluate_makespan(G, children, packet_size_per_subchunk, gpu_nodes):
    """
    Lightweight proxy for makespan of a single tree.
    Returns time in microseconds (float). This avoids calling the full simulator,
    so it can be used safely inside local search operators.
    """
    from decimal import Decimal
    # Recompute a root (node that never appears as a child)
    all_nodes = set(children.keys())
    for vs in children.values():
        all_nodes.update(vs)
    possible_parents = set(children.keys())
    possible_children = set(v for vs in children.values() for v in vs)
    roots = list(possible_parents - possible_children)
    root = roots[0] if roots else (next(iter(children.keys())) if children else None)
    if root is None:
        return 0.0
    # DFS longest downstream time using edge_cost (serialization + propagation)
    memo = {}
    def dfs(u):
        if u in memo:
            return memo[u]
        best = Decimal(0)
        for v in children.get(u, []):
            if not G.has_edge(u, v):
                continue
            ser = edge_cost(G, u, v, packet_size_per_subchunk)
            prop = G.edges[(u, v)]['propagation_latency']
            cand = ser + prop + dfs(v)
            if cand > best:
                best = cand
        memo[u] = best
        return best
    total = dfs(root)
    return float(total) * 1e6

def try_edge_exchange_two_opt(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes):
    """
    2-opt 边交换（TBR 的精简变体，触发边必须是 GPU–BR / BR–BR / BR–GPU）。
    选两条边 (a->b) 与 (c->d)，尝试替换为 (a->d) 与 (c->b)。
    要求：
      - a != c, b != d；
      - 至少有一条是触发边；
      - 交换后仍是树（无环、连通）；
      - 新边必须存在于拓扑且满足 BR / 跨 DC 约束。
    成功返回 (children, depth, maxd)，否则 None。
    """
    import random

    edges = [(u, v) for u, lst in children.items() for v in lst]
    if len(edges) < 2:
        print("[DEBUG-CAND][2-OPT] skip: less than 2 edges in tree")
        return None

    # 统计触发边数量（GPU->BR、BR->BR、BR->GPU）
    trigger_edges = [(u, v) for (u, v) in edges if is_trigger_edge(u, v, G)]
    print(f"[DEBUG-CAND][2-OPT] edges={len(edges)}, trigger_edges={len(trigger_edges)}")

    def attempt_swaps(require_trigger: bool):
        total_pairs = 0
        trigger_filtered = 0
        cycle_filtered = 0
        infeasible_after_swap = 0
        tried = 0

        e = list(edges)
        random.shuffle(e)
        trials = min(20, len(e) * (len(e) - 1) // 2)

        def do_swap(a, b, c, d):
            nonlocal infeasible_after_swap
            if not (G.has_edge(a, d) and edge_allowed_with_pair(G, a, d, br_pair)):
                infeasible_after_swap += 1
                return None
            if not (G.has_edge(c, b) and edge_allowed_with_pair(G, c, b, br_pair)):
                infeasible_after_swap += 1
                return None
            cand = {k: list(v) for k, v in children.items()}
            if b in cand.get(a, []):
                cand[a] = [x for x in cand[a] if x != b]
            if d in cand.get(c, []):
                cand[c] = [x for x in cand[c] if x != d]
            cand.setdefault(a, []).append(d)
            cand.setdefault(c, []).append(b)
            ok, clean, dmap, md = validate_tree_constraints(G, cand, src, br_pair, gpu_nodes)
            if not ok:
                infeasible_after_swap += 1
                return None
            # 仅返回候选，不在局部评估 makespan
            return clean, dmap, md

        for i in range(len(e)):
            if tried >= trials:
                break
            a, b = e[i]
            for j in range(i + 1, len(e)):
                c, d = e[j]
                total_pairs += 1
                if a == c or b == d:
                    trigger_filtered += 1
                    continue

                if require_trigger and not (is_trigger_edge(a, b, G) or is_trigger_edge(c, d, G)):
                    trigger_filtered += 1
                    continue

                # 防环：b 子树与 d 子树不能互相包含
                sub_b = collect_subtree_nodes(children, b)
                sub_d = collect_subtree_nodes(children, d)
                if a in sub_d or c in sub_b:
                    cycle_filtered += 1
                    continue

                res = do_swap(a, b, c, d)
                tried += 1
                if res is not None:
                    print(f"[DEBUG-CAND][2-OPT] success after {tried} attempts; pairs_considered={total_pairs}, trigger_filtered={trigger_filtered}, cycle_filtered={cycle_filtered}, infeasible={infeasible_after_swap}")
                    return res

        print(f"[DEBUG-CAND][2-OPT] no feasible candidate after {tried} attempted swaps; pairs_considered={total_pairs}, trigger_filtered={trigger_filtered}, cycle_filtered={cycle_filtered}, infeasible={infeasible_after_swap}")
        return None

    # 第一轮：严格触发
    res = attempt_swaps(require_trigger=True)
    if res is not None:
        # After a successful swap, log and return candidate
        # Find swapped edges for logging
        # We cannot know which swap was performed, but let's try to print the swap
        # The attempt_swaps returns (clean, dmap, md), but do_swap is called with (a,b,c,d)
        # To get the swapped edges, we can re-run attempt_swaps logic here for logging
        # Instead, let's replicate do_swap for logging, but since it's not accessible, we log with a generic message
        print(f"[DEBUG-CAND][2-OPT] generated candidate swapping ?")
        return res

    # 如果所有 pair 都因 trigger 被过滤，放宽一次（依旧由 validate_tree_constraints 把关）
    print("[DEBUG-CAND][2-OPT] relaxing trigger constraint for a second pass …")
    res = attempt_swaps(require_trigger=False)
    if res is not None:
        print(f"[DEBUG-CAND][2-OPT] generated candidate swapping ?")
        return res
    print(f"[DEBUG-CAND][2-OPT] no feasible candidate for src={src}")
    return None


# --- Additional topology-aware operators: insert / bypass / sibling graft ---

def try_insert_intermediate_node(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes):
    """
    在现有树的一条边 (u->v) 之间插入一个中间节点 m：
      将 (u->v) 替换为 (u->m) 和 (m->v)。
    约束：
      - G 中同时存在边 (u->m) 与 (m->v)，且 edge_allowed_with_pair(...) 为 True；
      - m 不能在 v 的子树里（避免环）；
      - 其余跨DC/BR/方向一致性由 validate_tree_constraints 检查。
    成功返回 (children, depth, maxd)，否则 None。
    """
    import random

    edges = [(u, v) for u, lst in children.items() for v in lst]
    if not edges:
        return None
    random.shuffle(edges)

    for (u, v) in edges:
        # trigger must be one of GPU–BR / BR–BR / BR–GPU
        if not is_trigger_edge(u, v, G):
            continue
        sub_v = collect_subtree_nodes(children, v)
        for m in G.successors(u):
            if m in sub_v or m == u or m == v:
                continue
            if not (G.has_edge(u, m) and G.has_edge(m, v)):
                continue
            if not (edge_allowed_with_pair(G, u, m, br_pair) and edge_allowed_with_pair(G, m, v, br_pair)):
                continue

            cand = {k: list(vs) for k, vs in children.items()}
            if v in cand.get(u, []):
                cand[u] = [x for x in cand[u] if x != v]
            cand.setdefault(u, []).append(m)
            cand.setdefault(m, []).append(v)

            ok, clean, dmap, md = validate_tree_constraints(G, cand, src, br_pair, gpu_nodes)
            if ok:
                print(f"[DEBUG-CAND][INSERT] generated candidate by inserting {m} between {u}->{v}")
                return clean, dmap, md
    print(f"[DEBUG-CAND][INSERT] no feasible candidate for src={src}")
    return None


def try_bypass_intermediate_node(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes):
    """
    对链式 u->m->v，如果存在直接边 u->v（且允许），则把 (u->m->v) 简化为 (u->v)。
    成功返回 (children, depth, maxd)，否则 None。
    """
    found = False
    for u, lst in children.items():
        for m in list(lst):
            for v in children.get(m, []):
                # require at least one of (u->m) or (m->v) to be a trigger edge
                if not (is_trigger_edge(u, m, G) or is_trigger_edge(m, v, G)):
                    continue
                if not (G.has_edge(u, v) and edge_allowed_with_pair(G, u, v, br_pair)):
                    continue
                cand = {k: list(vs) for k, vs in children.items()}
                if m in cand.get(u, []):
                    cand[u] = [x for x in cand[u] if x != m]
                if v in cand.get(m, []):
                    cand[m] = [x for x in cand[m] if x != v]
                cand.setdefault(u, []).append(v)
                ok, clean, dmap, md = validate_tree_constraints(G, cand, src, br_pair, gpu_nodes)
                if ok:
                    print(f"[DEBUG-CAND][BYPASS] generated candidate by bypassing {u}->{m}->{v} into {u}->{v}")
                    return clean, dmap, md
    print(f"[DEBUG-CAND][BYPASS] no feasible candidate for src={src}")
    return None


def try_graft_sibling(G, src, children, depth, br_pair, packet_size_per_subchunk, gpu_nodes):
    """
    兄弟重接：同父 u 的孩子 v、w 若存在允许的 v->w（或 w->v），
    则将 w(或v) 从 u 底下挂到 v(或w) 底下，缓解 u 的扇出竞争。
    成功返回 (children, depth, maxd)，否则 None。
    """
    import random
    parents = list(children.keys())
    random.shuffle(parents)
    for u in parents:
        kids = list(children.get(u, []))
        if len(kids) < 2:
            continue
        random.shuffle(kids)
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                v, w = kids[i], kids[j]

                # only allow graft if the triggering parent->child edge is of trigger type
                if not is_trigger_edge(u, w, G):
                    pass  # try the other direction below
                else:
                    if G.has_edge(v, w) and edge_allowed_with_pair(G, v, w, br_pair):
                        cand = {k: list(vs) for k, vs in children.items()}
                        if w in cand.get(u, []):
                            cand[u] = [x for x in cand[u] if x != w]
                        cand.setdefault(v, []).append(w)
                        ok, clean, dmap, md = validate_tree_constraints(G, cand, src, br_pair, gpu_nodes)
                        if ok:
                            print(f"[DEBUG-CAND][GRAFT] generated candidate by grafting {w} under {v}")
                            return clean, dmap, md

                # w -> v branch: only if the triggering parent->child edge is of trigger type
                if is_trigger_edge(u, v, G):
                    if G.has_edge(w, v) and edge_allowed_with_pair(G, w, v, br_pair):
                        cand = {k: list(vs) for k, vs in children.items()}
                        if v in cand.get(u, []):
                            cand[u] = [x for x in cand[u] if x != v]
                        cand.setdefault(w, []).append(v)
                        ok, clean, dmap, md = validate_tree_constraints(G, cand, src, br_pair, gpu_nodes)
                        if ok:
                            print(f"[DEBUG-CAND][GRAFT] generated candidate by grafting {v} under {w}")
                            return clean, dmap, md
    print(f"[DEBUG-CAND][GRAFT] no feasible candidate for src={src}")
    return None

def _longest_downstream_time_on_tree(G, children, root, packet_size_per_subchunk: Decimal):
    """返回：dict[node] = 从该节点到其叶子的最长串行传输时间（按 edge_cost）。"""
    memo = {}
    def dfs(u):
        if u in memo:
            return memo[u]
        best = Decimal(0)
        for v in children.get(u, []):
            c = edge_cost(G, u, v, packet_size_per_subchunk)
            best = max(best, c + dfs(v))
        memo[u] = best
        return best
    dfs(root)
    return memo

def is_trigger_edge(u, v, G):
    ut, vt = G.nodes[u].get("type"), G.nodes[v].get("type")
    # 原三类：GPU→BR、BR→BR、BR→GPU
    if ut == "GPU" and vt == "BR":
        return True
    if ut == "BR" and vt == "BR":
        return True
    if ut == "BR" and vt == "GPU":
        return True
    # 新增：同一 DC 内的 GPU→GPU
    if ut == "GPU" and vt == "GPU":
        dc_u = G.nodes[u].get("DC")
        dc_v = G.nodes[v].get("DC")
        if dc_u is not None and dc_v is not None and dc_u == dc_v:
            return True
    return False

# ---- Helper: Select slow subchunks for prioritization ----
import math

def select_slow_subchunks(per_best, min_k=3, delta=0.1):
    """
    根据每个 subchunk 的完成时间，选择需要优先优化的慢树。
    - per_best: dict[sub_id] -> 完成时间 (Decimal 或 None)
    - min_k: 至少选择多少棵树
    - delta: 偏离阈值 (如 0.1 表示比平均值高 10%)
    返回: list of sub_ids
    """
    from decimal import Decimal as _Dec

    if not per_best:
        return []

    vals = [t for t in per_best.values() if t is not None]
    if not vals:
        return []

    avg = sum(vals) / _Dec(len(vals))
    # 将 delta 转成 Decimal，避免 Decimal * float 的类型错误
    try:
        delta_dec = _Dec(str(delta))
    except Exception:
        delta_dec = _Dec('0')
    threshold = avg * (_Dec(1) + delta_dec)

    slow = [sid for sid, t in per_best.items() if (t is not None and t > threshold)]

    # 如果不足 min_k，就补齐为最慢的前 min_k 个
    if len(slow) < int(min_k):
        sorted_by_time = sorted(
            per_best.items(),
            key=lambda kv: (kv[1] if kv[1] is not None else _Dec('-Infinity')),
            reverse=True,
        )
        need = int(min_k)
        slow = [sid for sid, _ in sorted_by_time[:need]]

    return slow


def _pick_critical_subchunk_by_static(G, trees, depths, packet_size_per_subchunk: Decimal):
    """用静态 longest-path 近似从所有 subchunk 中挑一个最“慢”的（近似 critical）。"""
    worst_sid, worst_val = None, Decimal(-1)
    for sid, ch in trees.items():
        if sid not in depths or not depths[sid]:
            continue
        root_candidates = [n for n, d in depths[sid].items() if d == 0]
        if not root_candidates:
            continue
        root = root_candidates[0]
        down = _longest_downstream_time_on_tree(G, ch, root, packet_size_per_subchunk)
        val = down.get(root, Decimal(0))
        if val > worst_val:
            worst_val = val
            worst_sid = sid
    return worst_sid

def _reprioritize_bottlenecks_by_depth(G, packet_size_per_subchunk, gpu_nodes,
                                       trees, depths, max_depths, sub_src_of,
                                       iters: int = 3):
    """
    只调整“每个节点的孩子列表顺序”（即本地队列优先级），不改边：
      1) 跑一次流水线仿真，拿到每个 subchunk 完成时间 per_sub；
      2) 取完成时间 > 平均完成时间的 subchunk 作为瓶颈；
      3) 对这些 subchunk 的每个分叉节点，把“子树最大深度”更大的孩子排前；
      4) 迭代若干轮。
    """
    from decimal import Decimal

    def eval_combo(trees_, depths_, maxd_):
        ms_, per_ = simulate_allgather_pipeline_bfs(
            G, packet_size_per_subchunk, gpu_nodes,
            trees_override=trees_, depths_override=depths_, max_depths_override=maxd_,
            verbose=False
        )
        return ms_, per_

    def subtree_maxdepth(ch_map, depths_map, start_child):
        # 从 start_child 出发的子树上的最大 depth
        md = depths_map.get(start_child, 0)
        stack = [start_child]
        seen = {start_child}
        while stack:
            x = stack.pop()
            md = max(md, depths_map.get(x, md))
            for w in ch_map.get(x, []):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return md

    # build policy map once (if provided)
    pol_map = _policy_time_map(globals().get('policy', [])) if 'policy' in globals() else {}

    def policy_time_for(src_sid, u, v):
        src = sub_src_of[src_sid]
        cid = int(src) - 4
        return pol_map.get((cid, int(u), int(v)), None)


# ---- Helper: build a (chunk_id, u, v) -> Decimal(time) map from policy ----
# Ensure _policy_time_map is defined before any usage.
def _policy_time_map(policy_list):
    from decimal import Decimal
    p = {}
    for rec in policy_list:
        if not isinstance(rec, (list, tuple)) or len(rec) < 4:
            continue
        cid, u, v, t = rec[0], rec[1], rec[2], rec[3]
        try:
            tt = Decimal(str(t))
        except Exception:
            tt = Decimal('0')
        p[(int(cid), int(u), int(v))] = tt
    return p



def _evaluate_policy_baseline(G, packet_size_per_subchunk, gpu_nodes,
                              init_trees=None, init_depths=None, init_max_depths=None,
                              sub_src_of=None, policy_list=None):
    """
    Evaluate a policy-only baseline. If init_* trees/depths are provided, keep the edges but
    reorder each node's children according to the policy times (earlier -> higher priority).
    Otherwise, build trees from the policy if possible, or fall back to BFS trees.
    Returns: (makespan, per_sub_times, trees_ordered).
    """
    from decimal import Decimal

    # Build subchunk list/mapping if not provided
    if sub_src_of is None:
        num_chunks_per_gpu = 1
        num_subchunks_per_chunk = int(getattr(config, 'num_chunk', 1))
        sub_src_of = {}
        subchunks = []
        for src in gpu_nodes:
            for _c in range(num_chunks_per_gpu):
                for _k in range(num_subchunks_per_chunk):
                    sid = len(subchunks)
                    subchunks.append(sid)
                    sub_src_of[sid] = src
    else:
        subchunks = sorted(sub_src_of.keys())

    # Expose policy globally (simulator may bias queue ordering)
    try:
        global HOT_POLICY
        HOT_POLICY = policy_list or []
    except Exception:
        pass

    # Choose base trees/depths/max_depths
    trees = init_trees
    depths = init_depths
    max_depths = init_max_depths

    if trees is None or depths is None or max_depths is None:
        # Try to build from policy; else BFS fallback
        trees = depths = max_depths = None
        br_choice = {}
        if 'build_trees_from_policy' in globals() and policy_list:
            try:
                trees, depths, max_depths, br_choice = build_trees_from_policy(
                    G, subchunks, sub_src_of, packet_size_per_subchunk, gpu_nodes, policy_list
                )
            except Exception:
                trees = depths = max_depths = None
        if trees is None or depths is None or max_depths is None:
            trees, depths, max_depths = build_bfs_trees(G, subchunks, sub_src_of)

    # Reorder children by policy times (edges unchanged)
    try:
        pmap = _policy_time_map(policy_list or [])
        trees_ordered = _reorder_children_by_policy_map(trees, sub_src_of, pmap)
    except Exception:
        trees_ordered = {sid: {u: list(v) for u, v in ch.items()} for sid, ch in trees.items()}

    ms, per_sub = simulate_allgather_pipeline_bfs(
        G, packet_size_per_subchunk, gpu_nodes,
        trees_override=trees_ordered, depths_override=depths, max_depths_override=max_depths,
        verbose=False
    )
    return ms, per_sub, trees_ordered
# ========= Helper: Policy-based child ordering and tail-driven reprioritization =========

def _reorder_children_by_policy_map(trees, sub_src_of, policy_time_map):
    """
    Same as _reorder_children_by_policy but takes a prebuilt policy_time_map.
    """
    from decimal import Decimal
    new_trees = {sid: {u: list(vs) for u, vs in ch.items()} for sid, ch in trees.items()}
    for sid, ch in new_trees.items():
        src = sub_src_of[sid]
        chunk_id = int(src) - 4
        for u, lst in ch.items():
            if len(lst) <= 1:
                continue
            scored = []
            for v in lst:
                key = (chunk_id, int(u), int(v))
                t = policy_time_map.get(key, None)
                if t is None:
                    scored.append((1, Decimal('inf'), v))
                else:
                    scored.append((0, t, v))
            scored.sort(key=lambda x: (x[0], x[1], x[2]))
            ch[u] = [v for _, __, v in scored]
    return new_trees

def _internal_nodes(children):
    return [u for u,lst in children.items() if len(lst) >= 1]  # internal includes degree>=1 for ordering

def _bottleneck_reprioritize_policy(G, packet_size_per_subchunk, gpu_nodes,
                                    trees, depths, maxd, sub_src_of, policy,
                                    rounds: int = 6, alpha: 'Decimal|float' = 0.20):
    """
    Iteratively push tardy subchunks earlier on a few 'responsible' branching nodes by reducing their
    policy send times. This only changes child ordering (queue preference), never the tree edges.
    Heuristic for 'responsible nodes': branching nodes with the largest downstream time on that subchunk's tree.
    """
    from decimal import Decimal
    if not isinstance(alpha, Decimal):
        alpha = Decimal(str(alpha))

    # Build initial policy time map
    pmap = _policy_time_map(policy)
    best_ms = None
    best_trees = None
    best_per = None
    no_improve = 0

    for r in range(max(1, int(rounds))):
        # Apply current policy order
        trees_ord = _reorder_children_by_policy_map(trees, sub_src_of, pmap)
        ms, per = simulate_allgather_pipeline_bfs(
            G, packet_size_per_subchunk, gpu_nodes,
            trees_override=trees_ord, depths_override=depths, max_depths_override=maxd,
            verbose=False
        )
        if (best_ms is None) or (ms + Decimal('1e-18') < best_ms):
            best_ms, best_trees, best_per = ms, trees_ord, per
        # early stop if several consecutive rounds bring no improvement
        if best_ms is not None and (ms + Decimal('1e-18') >= best_ms):
            no_improve += 1
        else:
            no_improve = 0
        if no_improve >= 5:
            try:
                log(f"[EarlyStop] Tail-driven reprioritize: no improvement for {no_improve} consecutive rounds; stop.")
            except Exception:
                pass
            break

        # compute average completion
        vals = list(per.values())
        if not vals:
            break
        avg_t = sum(vals) / Decimal(len(vals))
        tardy = [(sid, per[sid] - avg_t) for sid in per if per[sid] > avg_t]
        if not tardy:
            break

        # Adjust priorities: for each tardy sid, pick a couple of heavy branching nodes
        for sid, tail in sorted(tardy, key=lambda kv: kv[1], reverse=True):
            ch = trees[sid]
            dmap = depths[sid]
            src = sub_src_of[sid]
            chunk_id = int(src) - 4

            # static downstream time per node for this sid
            down = _longest_downstream_time_on_tree(G, ch, src, packet_size_per_subchunk)
            # candidate nodes: internal / branching
            cand_nodes = [u for u in _internal_nodes(ch)]
            if not cand_nodes:
                continue
            # score by downstream time (bigger -> more critical)
            cand_nodes.sort(key=lambda u: ( -down.get(u, Decimal(0)), dmap.get(u, 10**9) ))

            # take top-2 nodes
            pick = cand_nodes[:2]
            for u in pick:
                # move this subchunk earlier on edges out of u: reduce its policy time on (u->child)
                for v in ch.get(u, []):
                    key = (chunk_id, int(u), int(v))
                    old = pmap.get(key, Decimal('1e9'))  # unknowns get very large default then lowered
                    new = max(Decimal('0'), old - alpha * tail)
                    pmap[key] = new

    return best_ms, best_trees, best_per, pmap

def _evaluate_combo_cached(G, packet_size_per_subchunk, gpu_nodes,
                           trees, depths, max_depths,
                           _cache):
    """对树组合求 makespan，带简单缓存。"""
    # 生成一个可哈希 key
    def key_for(trees):
        items = []
        for sid, ch in sorted(trees.items()):
            edges = []
            for u, lst in ch.items():
                for v in lst:
                    edges.append((u, v))
            edges.sort()
            items.append((sid, tuple(edges)))
        return tuple(items)
    K = key_for(trees)
    if K in _cache:
        return _cache[K]
    ms, per_sub = simulate_allgather_pipeline_bfs(
        G, packet_size_per_subchunk, gpu_nodes,
        trees_override=trees, depths_override=depths, max_depths_override=max_depths,
        verbose=False
    )
    _cache[K] = (ms, per_sub)
    return ms, per_sub


def lns_critical_beam(G, packet_size_per_subchunk: Decimal, gpu_nodes,
                      subchunks, sub_src_of,
                      start_trees, start_depths, start_max_depths, start_choice,
                      beam_width: int = 6, outer_iters: int = 25,
                      local_rewire_trials: int = 80):
    """更猛：基于 critical-subchunk 的大邻域搜索 + beam。
    步骤：
      1) 从当前组合里挑一个 *静态* 最慢的 subchunk（近似关键）。
      2) 生成邻域：
         - 该 subchunk 上尝试所有可行 BR 对（重建树）；
         - 在该 subchunk 上做多次 try_rewire_one（非贪心），从不同节点换父；
      3) 对每个新候选整体模拟，取 makespan 最小的前 B 个进入下一轮。
      4) 迭代多轮，返回全局最优。
    """
    print(">>> ENTERED lns_critical_beam <<<")
    try:
        log(f"[LNS] Start beam search with {len(start_trees)} trees, outer_iters={outer_iters}")
    except Exception:
        pass
    from decimal import Decimal
    # Early stopping: if no improvement for len(start_trees) consecutive outer iters, stop
    tree_count = len(start_trees) if start_trees is not None else 0
    no_improve = 0
    # 初始
    # best_trees = {k: {u: list(v) for u, v in start_trees.items()} for k, start_trees in start_trees.items()}  # deep copy
    # 上面一行写法较绕，改为更清晰：
    best_trees = {sid: {u: list(v) for u, v in ch.items()} for sid, ch in start_trees.items()}
    best_depths = {sid: dict(d) for sid, d in start_depths.items()}
    best_maxd = dict(start_max_depths)
    best_choice = dict(start_choice)

    cache = {}
    ms_best, per_sub = evaluate_with_edge_bottleneck_tuning(
        G, packet_size_per_subchunk, gpu_nodes,
        best_trees, best_depths, best_maxd,
        inner_rounds=3
    )

    # 预取所有可行 BR 对
    (_, _), _dc_to_brs, pairs = enumerate_br_pairs(G)

    beam = [(ms_best, best_trees, best_depths, best_maxd, best_choice)]

    for it in range(outer_iters):
        print(f"[DEBUG-LNS] Iter {it+1}/{outer_iters} started")
        prev_best_iter = ms_best
        next_pool = []
        seen = set()
        for ms_curr, trees_curr, depths_curr, maxd_curr, choice_curr in beam:
            # 1) 动态计算当前 candidate 的瓶颈 subchunks（完成时间 > 平均值）
            _ms_tmp, _per_tmp = evaluate_with_edge_bottleneck_tuning(
                G, packet_size_per_subchunk, gpu_nodes,
                trees_curr, depths_curr, maxd_curr,
                inner_rounds=1
            )
            # Use new helper to select slow subchunks
            slow_sids = select_slow_subchunks(_per_tmp, min_k=3, delta=0.1)
            # Fallback: if no slow_sids, use static critical sid
            if not slow_sids:
                sid_star = _pick_critical_subchunk_by_static(G, trees_curr, depths_curr, packet_size_per_subchunk)
                slow_sids = [sid_star if sid_star is not None else max(trees_curr.keys())]

            # 对每个瓶颈 subchunk 生成邻域
            for sid_target in slow_sids:
                src = sub_src_of[sid_target]

                # 2a) 试所有 BR 对
                for br_l, br_r in pairs:
                    if choice_curr.get(sid_target) == (br_l, br_r):
                        continue
                    res = build_tree_with_br_pair(G, src, br_l, br_r, packet_size_per_subchunk, gpu_nodes)
                    if res is None:
                        continue
                    ch, dep, md = res
                    cand_trees = {sid: {u: list(v) for u, v in chmap.items()} for sid, chmap in trees_curr.items()}
                    cand_depths = {sid: dict(d) for sid, d in depths_curr.items()}
                    cand_maxd = dict(maxd_curr)
                    cand_choice = dict(choice_curr)
                    cand_trees[sid_target] = ch
                    cand_depths[sid_target] = dep
                    cand_maxd[sid_target] = md
                    cand_choice[sid_target] = (br_l, br_r)

                    ms, _ = evaluate_with_edge_bottleneck_tuning(
                        G, packet_size_per_subchunk, gpu_nodes,
                        cand_trees, cand_depths, cand_maxd,
                        inner_rounds=2
                    )
                    key = (sid_target, br_l, br_r, 'br')
                    if key not in seen:
                        next_pool.append((ms, cand_trees, cand_depths, cand_maxd, cand_choice))
                        seen.add(key)

                # 2b) 多次非贪心换父
                tries = max(10, local_rewire_trials)
                for _ in range(tries):
                    trial = try_rewire_one(
                        G, src,
                        trees_curr[sid_target], depths_curr[sid_target],
                        choice_curr.get(sid_target), packet_size_per_subchunk, gpu_nodes,
                        greedy=False
                    )
                    if trial is None:
                        continue
                    ch, dep, md = trial
                    cand_trees = {sid: {u: list(v) for u, v in chmap.items()} for sid, chmap in trees_curr.items()}
                    cand_depths = {sid: dict(d) for sid, d in depths_curr.items()}
                    cand_maxd = dict(maxd_curr)
                    cand_choice = dict(choice_curr)

                    cand_trees[sid_target] = ch
                    cand_depths[sid_target] = dep
                    cand_maxd[sid_target] = md

                    ms, _ = evaluate_with_edge_bottleneck_tuning(
                        G, packet_size_per_subchunk, gpu_nodes,
                        cand_trees, cand_depths, cand_maxd,
                        inner_rounds=2
                    )
                    next_pool.append((ms, cand_trees, cand_depths, cand_maxd, cand_choice))

                # 2c) SPR chain
                for chain_len in (2, 3, 4):
                    trial_chain = try_rewire_chain(
                        G, src,
                        trees_curr[sid_target], depths_curr[sid_target],
                        choice_curr.get(sid_target), packet_size_per_subchunk, gpu_nodes,
                        steps=chain_len
                    )
                    if trial_chain is None:
                        continue
                    ch, dep, md = trial_chain
                    cand_trees = {sid: {u: list(v) for u, v in chmap.items()} for sid, chmap in trees_curr.items()}
                    cand_depths = {sid: dict(d) for sid, d in depths_curr.items()}
                    cand_maxd = dict(maxd_curr)
                    cand_choice = dict(choice_curr)
                    cand_trees[sid_target] = ch
                    cand_depths[sid_target] = dep
                    cand_maxd[sid_target] = md
                    ms, _ = evaluate_with_edge_bottleneck_tuning(
                        G, packet_size_per_subchunk, gpu_nodes,
                        cand_trees, cand_depths, cand_maxd,
                        inner_rounds=2
                    )
                    next_pool.append((ms, cand_trees, cand_depths, cand_maxd, cand_choice))

                # 2d) 2-opt
                for _ in range(12):
                    trial_2opt = try_edge_exchange_two_opt(
                        G, src,
                        trees_curr[sid_target], depths_curr[sid_target],
                        choice_curr.get(sid_target), packet_size_per_subchunk, gpu_nodes
                    )
                    if trial_2opt is None:
                        continue
                    ch, dep, md = trial_2opt
                    cand_trees = {sid: {u: list(v) for u, v in chmap.items()} for sid, chmap in trees_curr.items()}
                    cand_depths = {sid: dict(d) for sid, d in depths_curr.items()}
                    cand_maxd = dict(maxd_curr)
                    cand_choice = dict(choice_curr)
                    cand_trees[sid_target] = ch
                    cand_depths[sid_target] = dep
                    cand_maxd[sid_target] = md
                    ms, _ = evaluate_with_edge_bottleneck_tuning(
                        G, packet_size_per_subchunk, gpu_nodes,
                        cand_trees, cand_depths, cand_maxd,
                        inner_rounds=2
                    )
                    next_pool.append((ms, cand_trees, cand_depths, cand_maxd, cand_choice))

                # 2e) Insert intermediate
                if 'try_insert_intermediate_node' in globals():
                    for _ in range(12):
                        trial_ins = try_insert_intermediate_node(
                            G, src,
                            trees_curr[sid_target], depths_curr[sid_target],
                            choice_curr.get(sid_target), packet_size_per_subchunk, gpu_nodes
                        )
                        if trial_ins is None:
                            continue
                        ch, dep, md = trial_ins
                        cand_trees = {sid: {u: list(v) for u, v in chmap.items()} for sid, chmap in trees_curr.items()}
                        cand_depths = {sid: dict(d) for sid, d in depths_curr.items()}
                        cand_maxd = dict(maxd_curr)
                        cand_choice = dict(choice_curr)
                        cand_trees[sid_target] = ch
                        cand_depths[sid_target] = dep
                        cand_maxd[sid_target] = md
                        ms, _ = evaluate_with_edge_bottleneck_tuning(
                            G, packet_size_per_subchunk, gpu_nodes,
                            cand_trees, cand_depths, cand_maxd,
                            inner_rounds=2
                        )
                        next_pool.append((ms, cand_trees, cand_depths, cand_maxd, cand_choice))

                # 2f) Bypass intermediate
                if 'try_bypass_intermediate_node' in globals():
                    for _ in range(12):
                        trial_byp = try_bypass_intermediate_node(
                            G, src,
                            trees_curr[sid_target], depths_curr[sid_target],
                            choice_curr.get(sid_target), packet_size_per_subchunk, gpu_nodes
                        )
                        if trial_byp is None:
                            continue
                        ch, dep, md = trial_byp
                        cand_trees = {sid: {u: list(v) for u, v in chmap.items()} for sid, chmap in trees_curr.items()}
                        cand_depths = {sid: dict(d) for sid, d in depths_curr.items()}
                        cand_maxd = dict(maxd_curr)
                        cand_choice = dict(choice_curr)
                        cand_trees[sid_target] = ch
                        cand_depths[sid_target] = dep
                        cand_maxd[sid_target] = md
                        ms, _ = evaluate_with_edge_bottleneck_tuning(
                            G, packet_size_per_subchunk, gpu_nodes,
                            cand_trees, cand_depths, cand_maxd,
                            inner_rounds=2
                        )
                        next_pool.append((ms, cand_trees, cand_depths, cand_maxd, cand_choice))

                # 2g) Graft sibling
                if 'try_graft_sibling' in globals():
                    for _ in range(12):
                        trial_graft = try_graft_sibling(
                            G, src,
                            trees_curr[sid_target], depths_curr[sid_target],
                            choice_curr.get(sid_target), packet_size_per_subchunk, gpu_nodes
                        )
                        if trial_graft is None:
                            continue
                        ch, dep, md = trial_graft
                        cand_trees = {sid: {u: list(v) for u, v in chmap.items()} for sid, chmap in trees_curr.items()}
                        cand_depths = {sid: dict(d) for sid, d in depths_curr.items()}
                        cand_maxd = dict(maxd_curr)
                        cand_choice = dict(choice_curr)
                        cand_trees[sid_target] = ch
                        cand_depths[sid_target] = dep
                        cand_maxd[sid_target] = md
                        ms, _ = evaluate_with_edge_bottleneck_tuning(
                            G, packet_size_per_subchunk, gpu_nodes,
                            cand_trees, cand_depths, cand_maxd,
                            inner_rounds=2
                        )
                        next_pool.append((ms, cand_trees, cand_depths, cand_maxd, cand_choice))

        if not next_pool:
            break
        next_pool.sort(key=lambda x: x[0])
        beam = next_pool[:max(1, beam_width)]

        if beam[0][0] + Decimal('1e-18') < ms_best:
            ms_best, best_trees, best_depths, best_maxd, best_choice = \
                beam[0][0], beam[0][1], beam[0][2], beam[0][3], beam[0][4]
        print(f"[DEBUG-LNS] Iter {it+1}: current best makespan = {float(ms_best)*1e6:.3f} us")

        # Early stop check for tree adjustments (beam search)
        if ms_best + Decimal('1e-18') < prev_best_iter:
            no_improve = 0
        else:
            no_improve += 1
            if tree_count and no_improve >= tree_count:
                try:
                    log(f"[EarlyStop] LNS beam: no improvement for {no_improve} consecutive iterations (>= number of trees {tree_count}). Stop.")
                except Exception:
                    pass
                break

    print(f"[DEBUG-LNS] Finished with best makespan = {float(ms_best)*1e6:.3f} us")
    try:
        log(f"[LNS] Done: best makespan {float(ms_best)*1e6:.3f} us after {outer_iters} iterations")
    except Exception:
        pass
    return best_trees, best_depths, best_maxd, best_choice, ms_best


# ========= Hyper-heuristic + Path-Relinking (HHPR) =========


class _OperatorBandit:
    """
    Simple UCB1 bandit to adaptively pick mutation operators.
    Rewards are (old_ms - new_ms); we maximize expected improvement.
    """
    def __init__(self, ops):
        self.ops = ops
        self.count = {op: 0 for op in ops}
        self.value = {op: 0.0 for op in ops}
        self.total = 0

    def pick(self):
        import math, random
        self.total += 1
        # ensure each op tried at least once
        for op in self.ops:
            if self.count[op] == 0:
                return op
        # UCB
        logC = math.log(self.total)
        def ucb(op):
            avg = self.value[op] / self.count[op]
            return avg + (2.0 * (logC / self.count[op]) ) ** 0.5
        return max(self.ops, key=ucb)

    def update(self, op, reward):
        self.count[op] += 1
        self.value[op] += float(max(0.0, reward))

def hhpr_global_search(G, packet_size_per_subchunk,
                       gpu_nodes, subchunks, sub_src_of,
                       start_trees, start_depths, start_max_depths, start_choice,
                       generations=30, iters_per_gen=100, elite_size=8,
                       l2s_advisor=None, log_l2s_dataset=False, l2s_dataset_path=None):
    """
    Hyper-heuristic + (light) Path-Relinking search over the *combination* of trees.
    - Uses a UCB1 bandit to adaptively pick mutation operators.
    - Uses tardiness to pick which subchunk to perturb (or advisor-provided sid scores).
    - Each candidate is evaluated by evaluate_with_edge_bottleneck_tuning (i.e., with edge bottleneck scheduling).
    - At the end of each generation, perform a simple path-relinking / recombination among elites.
    Returns: (best_trees, best_depths, best_max_depths, best_choice, best_makespan)
    """
    print(">>> ENTERED hhpr_global_search <<<")
    try:
        log(f"[HHPR] Start global search with {len(start_trees)} trees, generations={generations}, iters/gen={iters_per_gen}")
    except Exception:
        pass
    from decimal import Decimal
    # Early stopping: if no improvement for len(start_trees) consecutive inner iterations, stop
    tree_count = len(start_trees) if start_trees is not None else 0
    no_improve = 0
    import random, math, json, os

    # --- Helper: biased pick of edge types for perturbation ---
    def _biased_pick_edge_type(sid, trees, depths):
        """
        在树中挑选一条边（或节点）作为优先修改目标，偏向 GPU–BR、BR–BR、BR–GPU 边。
        返回该 sid 的一个节点（u 或 v），用于后续 perturb。
        """
        import random
        ch = trees[sid]
        dmap = depths[sid]
        # 收集所有边
        gpu_br = []
        br_br = []
        br_gpu = []
        other = []
        for u, vs in ch.items():
            utype = G.nodes[u].get("type", "")
            for v in vs:
                vtype = G.nodes[v].get("type", "")
                # GPU–BR
                if utype == "GPU" and vtype == "BR":
                    gpu_br.append((u, v))
                # BR–BR
                elif utype == "BR" and vtype == "BR":
                    br_br.append((u, v))
                # BR–GPU
                elif utype == "BR" and vtype == "GPU":
                    br_gpu.append((u, v))
                else:
                    other.append((u, v))
        # 以高概率优先三类边
        p_focus = 0.7
        focus_edges = gpu_br + br_br + br_gpu
        if focus_edges and random.random() < p_focus:
            u, v = random.choice(focus_edges)
            # 返回被修改目标节点（如子节点v）
            return v
        elif other:
            u, v = random.choice(other)
            return v
        # fallback: 随机节点
        all_nodes = list(ch.keys())
        if all_nodes:
            return random.choice(all_nodes)
        return None

    # --- Helper: apply a local operator to a single subchunk tree ---
    def _apply_op(op, sid, trees, depths, maxd, choice, br_pairs):
        src = sub_src_of[sid]
        if op == 'change_br':
            # change to a different feasible BR pair
            for _ in range(10):
                br_l, br_r = random.choice(br_pairs)
                if choice.get(sid) == (br_l, br_r):
                    continue
                res = build_tree_with_br_pair(G, src, br_l, br_r, packet_size_per_subchunk, gpu_nodes)
                if res is None:
                    continue
                ch, dep, md = res
                new_trees = {k: {u: list(v) for u, v in val.items()} for k, val in trees.items()}
                new_depths = {k: dict(v) for k, v in depths.items()}
                new_maxd = dict(maxd)
                new_choice = dict(choice)
                new_trees[sid] = ch
                new_depths[sid] = dep
                new_maxd[sid] = md
                new_choice[sid] = (br_l, br_r)
                return new_trees, new_depths, new_maxd, new_choice
            return None

        if op == 'rewire_one':
            trial = try_rewire_one(G, src, trees[sid], depths[sid], choice.get(sid),
                                   packet_size_per_subchunk, gpu_nodes, greedy=False)
        elif op == 'rewire_chain':
            chain_len = random.choice((2, 3, 4))
            trial = try_rewire_chain(G, src, trees[sid], depths[sid], choice.get(sid),
                                     packet_size_per_subchunk, gpu_nodes, steps=chain_len)
        elif op == 'two_opt':
            trial = try_edge_exchange_two_opt(G, src, trees[sid], depths[sid], choice.get(sid),
                                              packet_size_per_subchunk, gpu_nodes)
        elif op == 'insert_mid':
            trial = try_insert_intermediate_node(G, src, trees[sid], depths[sid], choice.get(sid),
                                                 packet_size_per_subchunk, gpu_nodes)
        elif op == 'bypass_mid':
            trial = try_bypass_intermediate_node(G, src, trees[sid], depths[sid], choice.get(sid),
                                                 packet_size_per_subchunk, gpu_nodes)
        elif op == 'graft_sibling':
            trial = try_graft_sibling(G, src, trees[sid], depths[sid], choice.get(sid),
                                      packet_size_per_subchunk, gpu_nodes)
        else:
            trial = None

        if trial is None:
            return None
        ch, dep, md = trial
        new_trees = {k: {u: list(v) for u, v in val.items()} for k, val in trees.items()}
        new_depths = {k: dict(v) for k, v in depths.items()}
        new_maxd = dict(maxd)
        new_choice = dict(choice)
        new_trees[sid] = ch
        new_depths[sid] = dep
        new_maxd[sid] = md
        return new_trees, new_depths, new_maxd, new_choice

    # --- Helper: build simple state features for advisor/bandit ---
    def _build_state_feats(ms_curr, per_curr, max_depths):
        vals = list(per_curr.values()) if per_curr else []
        if vals:
            avg_t = sum(vals) / Decimal(len(vals))
            tardy = {int(s): float(per_curr[s] - avg_t) for s in per_curr if per_curr[s] > avg_t}
        else:
            avg_t = Decimal(0)
            tardy = {}
        sfeat = {
            "num_nodes": len(G.nodes),
            "num_edges": len(G.edges),
            "tardy": tardy,
            "max_depths": {int(k): int(v) for k, v in (max_depths or {}).items()},
            "makespan": float(ms_curr),
        }
        return sfeat, tardy

    # --- Helper: choose sid (subchunk id) to mutate ---
    def _pick_sid(tardy_map, sid_scores):
        if sid_scores and isinstance(sid_scores, dict) and len(sid_scores) > 0:
            # pick highest scored sid with some randomness
            sids_sorted = sorted(sid_scores.items(), key=lambda kv: kv[1], reverse=True)
            cut = min(3, len(sids_sorted))
            return int(random.choice([s for s, _ in sids_sorted[:cut]]))
        if tardy_map:
            # pick the most tardy
            return int(max(tardy_map.items(), key=lambda kv: kv[1])[0])
        return random.choice(subchunks)

    # --- Precompute feasible BR pairs ---
    (_, _), _dc_to_brs, br_pairs = enumerate_br_pairs(G)

    # --- Operators (expanded) ---
    ops = ['change_br', 'rewire_one', 'rewire_chain', 'two_opt', 'insert_mid', 'bypass_mid', 'graft_sibling']
    bandit = _OperatorBandit(ops)

    # --- Base candidate ---
    curr_trees = {sid: {u: list(v) for u, v in start_trees.items()} for sid, start_trees in start_trees.items()}
    curr_depths = {sid: dict(d) for sid, d in start_depths.items()}
    curr_maxd = dict(start_max_depths)
    curr_choice = dict(start_choice)

    ms_curr, per_curr = evaluate_with_edge_bottleneck_tuning(
        G, packet_size_per_subchunk, gpu_nodes,
        curr_trees, curr_depths, curr_maxd,
        inner_rounds=2
    )

    best_ms = ms_curr
    best_trees = {sid: {u: list(v) for u, v in ch.items()} for sid, ch in curr_trees.items()}
    best_depths = {sid: dict(d) for sid, d in curr_depths.items()}
    best_maxd = dict(curr_maxd)
    best_choice = dict(curr_choice)

    # --- Elite pool (keyed by edge set signature) ---
    def _key_from(trees):
        items = []
        for sid, ch in sorted(trees.items()):
            edges = []
            for u, lst in ch.items():
                for v in lst:
                    edges.append((u, v))
            edges.sort()
            items.append((sid, tuple(edges)))
        return tuple(items)

    elites = [ (best_ms, best_trees, best_depths, best_maxd, best_choice) ]
    elite_keys = { _key_from(best_trees) }

    # Dataset logging init
    if log_l2s_dataset and l2s_dataset_path:
        try:
            os.makedirs(os.path.dirname(l2s_dataset_path), exist_ok=True)
        except Exception:
            pass

    # --- Main search loop ---
    for gen in range(max(1, int(generations))):
        print(f"[DEBUG-HHPR] Generation {gen+1}/{generations} started")
        # inner loop (mutations)
        for it in range(max(1, int(iters_per_gen))):
            prev_best_global = best_ms
            # 根据当前解的瓶颈集合批量尝试：每个瓶颈 sid 调整一次
            state_feats, tardy_map = _build_state_feats(ms_curr, per_curr, curr_maxd)
            # target_sids: 按 tardy_map 排序
            target_sids = sorted(tardy_map.keys(), key=lambda s: tardy_map[s], reverse=True) if tardy_map else [random.choice(subchunks)]

            # 用 _biased_pick_edge_type 决定 perturb 目标边类别
            # 这里实现为：对 tardy sid，优先 perturb 重点边类别
            if target_sids:
                # 只对第一个 tardy sid 使用偏向边类别，其他保持原逻辑
                sid_main = target_sids[0]
                biased_node = _biased_pick_edge_type(sid_main, curr_trees, curr_depths)
                # 可以将 sid_main 替换为该节点所在 sid（这里 sid_main 即 tardy sid），也可将 perturb 目标用于 operator 内部
                # 这里保留 sid 但可在 operator 内部用 biased_node
                target_sids = [sid_main] + [s for s in target_sids[1:]]

            for sid in target_sids:
                # 选择算子（先问 advisor，再退化到 bandit）
                if l2s_advisor is not None:
                    try:
                        op_probs, sid_scores = l2s_advisor.predict(state_feats)
                    except Exception:
                        op_probs, sid_scores = ({}, None)
                else:
                    op_probs, sid_scores = ({}, None)

                if op_probs:
                    ops_list, probs = zip(*[(o, max(0.0, float(p))) for o, p in op_probs.items() if o in ops])
                    s = sum(probs) or 1.0
                    r = random.random()
                    acc = 0.0
                    picked_op = ops_list[-1]
                    for o, p in zip(ops_list, probs):
                        acc += p / s
                        if r <= acc:
                            picked_op = o
                            break
                else:
                    picked_op = bandit.pick()

                # 可以在 operator 内部根据 biased_node 进一步实现针对性 perturb
                res = _apply_op(picked_op, sid, curr_trees, curr_depths, curr_maxd, curr_choice, br_pairs)
                if res is None:
                    bandit.update(picked_op, -1e-6)
                    continue
                cand_trees, cand_depths, cand_maxd, cand_choice = res

                ms_new, per_new = evaluate_with_edge_bottleneck_tuning(
                    G, packet_size_per_subchunk, gpu_nodes,
                    cand_trees, cand_depths, cand_maxd,
                    inner_rounds=2
                )

                reward = float(ms_curr - ms_new)
                bandit.update(picked_op, reward)

                # 接受准则（改为 tardiness 方差自适应温度）
                accept = False
                if ms_new + Decimal('1e-18') < ms_curr:
                    accept = True
                else:
                    vals = list(per_new.values()) if per_new else []
                    var_tardy = float(sum((float(x - sum(vals)/len(vals)))**2 for x in vals)) / (len(vals) or 1) if vals else 1.0
                    T = max(1e-12, var_tardy * 0.1)
                    prob = math.exp(-max(0.0, float(ms_new - ms_curr)) / T)
                    if random.random() < prob:
                        accept = True

                if accept:
                    curr_trees, curr_depths, curr_maxd, curr_choice = cand_trees, cand_depths, cand_maxd, cand_choice
                    ms_curr, per_curr = ms_new, per_new
                    if ms_curr + Decimal('1e-18') < best_ms:
                        best_ms = ms_curr
                        best_trees = {sid_: {u: list(v) for u, v in ch.items()} for sid_, ch in curr_trees.items()}
                        best_depths = {sid_: dict(d) for sid_, d in curr_depths.items()}
                        best_maxd = dict(curr_maxd)
                        best_choice = dict(curr_choice)
                print(f"[DEBUG-HHPR] Gen {gen+1}, Iter {it+1}: current best makespan = {float(best_ms)*1e6:.3f} us")

            # Early stop check for tree adjustments (HHPR)
            if best_ms + Decimal('1e-18') < prev_best_global:
                no_improve = 0
            else:
                no_improve += 1
                if tree_count and no_improve >= tree_count:
                    try:
                        log(f"[EarlyStop] HHPR: no improvement for {no_improve} consecutive attempts (>= number of trees {tree_count}). Stop.")
                    except Exception:
                        pass
                    return best_trees, best_depths, best_maxd, best_choice, best_ms

            # maintain elites
            if 'cand_trees' not in locals() or cand_trees is None:
                continue
            k = _key_from(cand_trees)
            if (ms_new + Decimal('1e-18') < elites[-1][0] if len(elites) >= elite_size else True) and k not in elite_keys:
                elites.append((ms_new, cand_trees, cand_depths, cand_maxd, cand_choice))
                elite_keys.add(k)
                elites.sort(key=lambda x: x[0])
                if len(elites) > elite_size:
                    drop = elites.pop()
                    # remove key of dropped
                    try:
                        dk = _key_from(drop[1])
                        if dk in elite_keys:
                            elite_keys.remove(dk)
                    except Exception:
                        pass

        # --- Path-relinking among top-2 elites (light) ---
        if len(elites) >= 2:
            pa = elites[0]
            pb = elites[1]
            child_trees, child_depths, child_maxd, child_choice = _crossover_mix_trees(
                G, packet_size_per_subchunk, gpu_nodes,
                pa[1], pb[1], pa[2], pb[2], pa[3], pb[3], pa[4], pb[4]
            )
            ms_child, _ = evaluate_with_edge_bottleneck_tuning(
                G, packet_size_per_subchunk, gpu_nodes,
                child_trees, child_depths, child_maxd,
                inner_rounds=2
            )
            if ms_child + Decimal('1e-18') < best_ms:
                best_ms = ms_child
                best_trees, best_depths, best_maxd, best_choice = child_trees, child_depths, child_maxd, child_choice
            elites.append((ms_child, child_trees, child_depths, child_maxd, child_choice))
            elites.sort(key=lambda x: x[0])
            if len(elites) > elite_size:
                elites = elites[:elite_size]
                elite_keys = { _key_from(e[1]) for e in elites }

        # restart from current best elite to stabilize next generation
        ms_curr, curr_trees, curr_depths, curr_maxd, curr_choice = elites[0][0], elites[0][1], elites[0][2], elites[0][3], elites[0][4]
        # and recompute per_curr for the chosen start
        _, per_curr = evaluate_with_edge_bottleneck_tuning(
            G, packet_size_per_subchunk, gpu_nodes,
            curr_trees, curr_depths, curr_maxd,
            inner_rounds=1
        )

    try:
        log(f"[HHPR] Done: best makespan {float(best_ms)*1e6:.3f} us after {generations} generations")
    except Exception:
        pass
    print(f"[DEBUG-HHPR] Finished with best makespan = {float(best_ms)*1e6:.3f} us")
    return best_trees, best_depths, best_maxd, best_choice, best_ms

# ========= Learn-to-Search (GNN+RL) Hooks (torch-free fallback) =========
from typing import Optional

def _featurize_l2s(state_feats):
    """Torch-free featurizer and sid scoring (uses tardiness)."""
    num_nodes = float(state_feats.get('num_nodes', 0))
    num_edges = float(state_feats.get('num_edges', 0))
    hist = state_feats.get('edge_use_hist', {}) or {}
    h_len = float(hist.get('len', 0))
    h_p50 = float(hist.get('p50', 0))
    h_max = float(hist.get('max', 0))
    h_sum = float(hist.get('sum', 0))
    tardy = state_feats.get('tardy', {}) or {}
    t_vals = list(tardy.values())
    if t_vals:
        t_min = float(min(t_vals)); t_max = float(max(t_vals))
        t_mean = float(sum(t_vals) / len(t_vals))
    else:
        t_min = t_max = t_mean = 0.0
    md = state_feats.get('max_depths', {}) or {}
    d_vals = list(md.values())
    if d_vals:
        d_min = float(min(d_vals)); d_max = float(max(d_vals))
        d_mean = float(sum(d_vals) / max(1, len(d_vals)))
    else:
        d_min = d_max = d_mean = 0.0
    vec = [num_nodes, num_edges, h_len, h_p50, h_max, h_sum, t_min, t_mean, t_max, d_min, d_mean, d_max]
    sid_scores = {int(k): float(v) for k, v in tardy.items()} if tardy else None
    return vec, sid_scores

# Optional graph converters — keep API, return None
def state_to_pyg_graph(G, state_feats):  # noqa: F401
    return None

def state_to_dgl_graph(G, state_feats):  # noqa: F401
    return None

class L2SAdvisor:
    """Torch-free placeholder with the same API; returns uniform op probs and tardy-based sid scores."""
    def __init__(self, op_list=None, model_path: Optional[str] = None, device: Optional[str] = None, temperature: float = 1.0, seed: Optional[int] = 42):
        self.op_list = op_list or ['change_br', 'rewire_one', 'rewire_chain', 'two_opt', 'insert_mid', 'bypass_mid', 'graft_sibling']
        self.temperature = max(1e-6, float(temperature))
        self.device = 'cpu'
        self.model = None

    def predict(self, state_feats):
        _, sid_scores = _featurize_l2s(state_feats)
        if not self.op_list:
            return ({}, sid_scores)
        p = 1.0 / float(len(self.op_list))
        return ({op: p for op in self.op_list}, sid_scores)
# ========= end of Learn-to-Search (GNN+RL) Hooks (torch-free) =========

def _combo_key_from_trees(trees):
    items = []
    for sid, ch in sorted(trees.items()):
        edges = []
        for u, lst in ch.items():
            for v in lst:
                edges.append((u, v))
        edges.sort()
        items.append((sid, tuple(edges)))
    return tuple(items)

def _combo_hamming_distance(trees_a, trees_b):
    """Diversity metric between two combos: number of differing edges across all subchunks."""
    set_a = set()
    for sid, ch in trees_a.items():
        for u, lst in ch.items():
            for v in lst:
                set_a.add((sid, u, v))
    set_b = set()
    for sid, ch in trees_b.items():
        for u, lst in ch.items():
            for v in lst:
                set_b.add((sid, u, v))
    return len(set_a.symmetric_difference(set_b))

def _crossover_mix_trees(G, packet_size_per_subchunk, gpu_nodes,
                         parentA, parentB, depthsA, depthsB, maxdA, maxdB, choiceA, choiceB):
    """
    Recombine two combos, subchunk-wise:
      - For each subchunk, pick parent's tree with 50/50, then do one feasibility repair via validate_tree_constraints;
      - If chosen tree violates constraints (shouldn't, but in case of drift), fall back to the other parent.
    """
    import random
    child_trees, child_depths, child_maxd, child_choice = {}, {}, {}, {}
    for sid in parentA.keys():
        if random.random() < 0.5:
            ch, dmap, md, ch_pair = parentA[sid], depthsA[sid], maxdA[sid], choiceA.get(sid)
        else:
            ch, dmap, md, ch_pair = parentB[sid], depthsB[sid], maxdB[sid], choiceB.get(sid)
        # Quick feasibility check
        # root:
        roots = [n for n, dd in dmap.items() if dd == 0]
        if not roots:
            # fallback
            ch, dmap, md, ch_pair = parentA[sid], depthsA[sid], maxdA[sid], choiceA.get(sid)
        else:
            src = roots[0]
            ok, clean, dnew, mdnew = validate_tree_constraints(G, ch, src, ch_pair, gpu_nodes)
            if ok:
                ch, dmap, md = clean, dnew, mdnew
            else:
                ch, dmap, md, ch_pair = parentA[sid], depthsA[sid], maxdA[sid], choiceA.get(sid)
        child_trees[sid] = {u: list(v) for u, v in ch.items()}
        child_depths[sid] = dict(dmap)
        child_maxd[sid] = md
        child_choice[sid] = ch_pair
    return child_trees, child_depths, child_maxd, child_choice

def _mutate_one_tree(G, packet_size_per_subchunk, gpu_nodes,
                     sid, src, trees, depths, maxd, choice,
                     br_pairs, p_br=0.3, p_rewire=0.5, p_twoopt=0.2, rewire_trials=3):
    """
    Apply one of several local mutations to a single subchunk tree while respecting constraints:
      - Change BR pair (rebuild with build_tree_with_br_pair)
      - Single-step rewire (try_rewire_one greedy=False)
      - 2-opt edge exchange (try_edge_exchange_two_opt)
    Returns possibly-updated (trees, depths, maxd, choice). If no valid change found, returns originals.
    """
    import random
    r = random.random()
    if r < p_br and br_pairs:
        # Change BR pair
        for _ in range(6):
            br_l, br_r = random.choice(br_pairs)
            if choice.get(sid) == (br_l, br_r):
                continue
            res = build_tree_with_br_pair(G, src, br_l, br_r, packet_size_per_subchunk, gpu_nodes)
            if res is None:
                continue
            ch, dep, md = res
            new_trees = {k: {u: list(v) for u, v in val.items()} for k, val in trees.items()}
            new_depths = {k: dict(v) for k, v in depths.items()}
            new_maxd = dict(maxd)
            new_choice = dict(choice)
            new_trees[sid] = ch
            new_depths[sid] = dep
            new_maxd[sid] = md
            new_choice[sid] = (br_l, br_r)
            return new_trees, new_depths, new_maxd, new_choice
        return trees, depths, maxd, choice

    if r < p_br + p_rewire:
        # Try several rewires, pick the first feasible candidate
        for _ in range(max(1, rewire_trials)):
            trial = try_rewire_one(G, src, trees[sid], depths[sid], choice.get(sid),
                                   packet_size_per_subchunk, gpu_nodes, greedy=False)
            if trial is None:
                continue
            ch, dep, md = trial
            new_trees = {k: {u: list(v) for u, v in val.items()} for k, val in trees.items()}
            new_depths = {k: dict(v) for k, v in depths.items()}
            new_maxd = dict(maxd)
            new_choice = dict(choice)
            new_trees[sid] = ch
            new_depths[sid] = dep
            new_maxd[sid] = md
            return new_trees, new_depths, new_maxd, new_choice
        return trees, depths, maxd, choice

    # Two-opt
    trial2 = try_edge_exchange_two_opt(G, src, trees[sid], depths[sid], choice.get(sid),
                                       packet_size_per_subchunk, gpu_nodes)
    if trial2 is not None:
        ch, dep, md = trial2
        new_trees = {k: {u: list(v) for u, v in val.items()} for k, val in trees.items()}
        new_depths = {k: dict(v) for k, v in depths.items()}
        new_maxd = dict(maxd)
        new_choice = dict(choice)
        new_trees[sid] = ch
        new_depths[sid] = dep
        new_maxd[sid] = md
        return new_trees, new_depths, new_maxd, new_choice

    return trees, depths, maxd, choice

def optimize_tree_combo_pbm(G, packet_size_per_subchunk: Decimal, gpu_nodes,
                            subchunks, sub_src_of,
                            init_trees, init_depths, init_max_depths, init_choice,
                            pop_size: int = 10, generations: int = 25,
                            elite_keep: int = 3, mutation_per_iter: int = 4):
    """
    Population-based memetic search over the *combination* (all trees together).
    - Maintains a population of combos (trees+depths+maxd+choice).
    - Each generation: selection (top-k + diversity), crossover, mutations guided by critical subchunks.
    - Each candidate is globally evaluated by full pipeline simulation (makespan).
    """
    import random
    from decimal import Decimal
    # Early stopping: if no improvement for len(init_trees) consecutive generations, stop
    tree_count = len(init_trees) if init_trees is not None else 0
    no_improve = 0

    # Cache to avoid repeated sims
    cache = {}

    def eval_combo(trees, depths, maxd):
        K = _combo_key_from_trees(trees)
        if K in cache:
            return cache[K]
        ms, per_sub = evaluate_with_edge_bottleneck_tuning(
            G, packet_size_per_subchunk, gpu_nodes,
            trees, depths, maxd,
            inner_rounds=2
        )
        cache[K] = (ms, per_sub)
        return ms, per_sub

    # All feasible BR pairs
    (_, _), _dc_to_brs, br_pairs = enumerate_br_pairs(G)

    # Seed population from local improvements of the init
    base_ms, base_per = eval_combo(init_trees, init_depths, init_max_depths)
    population = [(base_ms, init_trees, init_depths, init_max_depths, init_choice, base_per)]

    # add a few BR-perturbed variants
    while len(population) < pop_size:
        trees = {sid: {u: list(v) for u, v in ch.items()} for sid, ch in init_trees.items()}
        depths = {sid: dict(d) for sid, d in init_depths.items()}
        maxd = dict(init_max_depths)
        choice = dict(init_choice)
        # randomly mutate a few subchunks
        for _ in range(3):
            sid = random.choice(subchunks)
            src = sub_src_of[sid]
            trees, depths, maxd, choice = _mutate_one_tree(
                G, packet_size_per_subchunk, gpu_nodes,
                sid, src, trees, depths, maxd, choice,
                br_pairs, rewire_trials=2
            )
        ms, per = eval_combo(trees, depths, maxd)
        population.append((ms, trees, depths, maxd, choice, per))

    # Track best so far for early stopping
    best_so_far = None
    # Evolution
    for gen in range(generations):
        # Sort by makespan
        population.sort(key=lambda x: x[0])

        # Elites
        elites = population[:elite_keep]

        # Diversity-aware parent pool (top half + far ones)
        parent_pool = population[:max(elite_keep, pop_size//2)]
        # add some diverse outliers
        for cand in population[elite_keep:]:
            far = True
            for (_, tE, _, _, _, _) in elites:
                if _combo_hamming_distance(cand[1], tE) <= 4:
                    far = False
                    break
            if far:
                parent_pool.append(cand)
            if len(parent_pool) >= pop_size:
                break

        # Offspring set
        offsprings = []
        # Crossover
        for _ in range(max(1, pop_size - elite_keep)):
            pa = random.choice(parent_pool)
            pb = random.choice(parent_pool)
            # Combine
            child_trees, child_depths, child_maxd, child_choice = _crossover_mix_trees(
                G, packet_size_per_subchunk, gpu_nodes,
                pa[1], pb[1], pa[2], pb[2], pa[3], pb[3], pa[4], pb[4]
            )
            # Guided mutations on *critical* subchunks (those with largest completion time in parents)
            # Use the worse parent per-sub as guidance
            worst_per = pa[5]
            if pb[0] > pa[0]:
                worst_per = pb[5]
            # pick a few tardy subchunks
            tardy = sorted(worst_per.items(), key=lambda kv: kv[1], reverse=True)
            choose = [sid for sid, _ in tardy[:mutation_per_iter]] if tardy else subchunks[:mutation_per_iter]
            for sid in choose:
                src = sub_src_of[sid]
                child_trees, child_depths, child_maxd, child_choice = _mutate_one_tree(
                    G, packet_size_per_subchunk, gpu_nodes,
                    sid, src,
                    child_trees, child_depths, child_maxd, child_choice,
                    br_pairs, rewire_trials=3
                )
            ms, per = eval_combo(child_trees, child_depths, child_maxd)
            offsprings.append((ms, child_trees, child_depths, child_maxd, child_choice, per))

        # New population = elites + best offspring
        combined = elites + offsprings
        combined.sort(key=lambda x: x[0])
        population = combined[:pop_size]

        # Early stop per generation when no global improvement
        population.sort(key=lambda x: x[0])
        curr_best = population[0][0]
        if (best_so_far is None) or (curr_best + Decimal('1e-18') < best_so_far):
            best_so_far = curr_best
            no_improve = 0
        else:
            no_improve += 1
            if tree_count and no_improve >= tree_count:
                try:
                    log(f"[EarlyStop] PBMS: no improvement for {no_improve} consecutive generations (>= number of trees {tree_count}). Stop.")
                except Exception:
                    pass
                break

    # Return best
    population.sort(key=lambda x: x[0])
    ms_best, bt, bd, bm, bc, _ = population[0]
    return bt, bd, bm, bc, ms_best


import time
# ========= 入口：无需命令行 =========
if __name__ == "__main__":
    t_start = time.time()
    from decimal import Decimal
    # --- 配置参数（与拓扑一致）---
    config.packet_size = Decimal(str(1/1024))      # 总包大小（将被 NVD2_1_topology 用 / num_chunk）
    config.num_chunk = getattr(config, 'num_chunk', 1)  # subchunk 个数
    config.chassis = 2
    config.collective = 'ALLGATHER'
    config.topology_name = 'NVD2'
    config.connect_matrix = []
    config.connectivity = 0.5

    policy = [[0, 4, 0, Decimal('0.0')], [0, 4, 5, Decimal('0.0')], [0, 4, 6, Decimal('0.0')], [0, 4, 8, Decimal('0.0')], [1, 5, 1, Decimal('0.0')], [1, 5, 4, Decimal('0.0')], [1, 5, 6, Decimal('0.0')], [2, 6, 4, Decimal('0.0')], [2, 6, 5, Decimal('0.0')], [2, 6, 7, Decimal('0.0')], [3, 7, 0, Decimal('0.0')], [3, 7, 6, Decimal('0.0')], [3, 7, 8, Decimal('0.0')], [4, 8, 0, Decimal('0.0')], [4, 8, 4, Decimal('0.0')], [4, 8, 7, Decimal('0.0')], [4, 8, 9, Decimal('0.0')], [5, 9, 1, Decimal('0.0')], [5, 9, 8, Decimal('0.0')], [6, 10, 2, Decimal('0.0')], [6, 10, 11, Decimal('0.0')], [7, 11, 3, Decimal('0.0')], [7, 11, 10, Decimal('0.0')], [7, 11, 12, Decimal('0.0')], [7, 11, 15, Decimal('0.0')], [8, 12, 2, Decimal('0.0')], [8, 12, 11, Decimal('0.0')], [8, 12, 13, Decimal('0.0')], [9, 13, 2, Decimal('0.0')], [9, 13, 12, Decimal('0.0')], [9, 13, 14, Decimal('0.0')], [9, 13, 15, Decimal('0.0')], [10, 14, 13, Decimal('0.0')], [10, 14, 15, Decimal('0.0')], [11, 15, 3, Decimal('0.0')], [11, 15, 11, Decimal('0.0')], [11, 15, 13, Decimal('0.0')], [11, 15, 14, Decimal('0.0')], [4, 0, 2, Decimal('7E-7')], [3, 0, 3, Decimal('7E-7')], [5, 1, 2, Decimal('7E-7')], [1, 1, 3, Decimal('7E-7')], [9, 2, 0, Decimal('7E-7')], [8, 2, 1, Decimal('7E-7')], [11, 3, 0, Decimal('7E-7')], [7, 3, 1, Decimal('7E-7')], [2, 4, 1, Decimal('0.0000397625')], [4, 4, 5, Decimal('0.0000397625')], [4, 4, 6, Decimal('0.0000397625')], [1, 4, 8, Decimal('0.0000397625')], [3, 6, 4, Decimal('0.0000397625')], [3, 6, 5, Decimal('0.0000397625')], [1, 6, 7, Decimal('0.0000397625')], [2, 7, 8, Decimal('0.0000397625')], [5, 8, 4, Decimal('0.0000397625')], [5, 8, 7, Decimal('0.0000397625')], [0, 8, 9, Decimal('0.0000397625')], [8, 11, 10, Decimal('0.0000397625')], [6, 11, 12, Decimal('0.0000397625')], [6, 11, 15, Decimal('0.0000397625')], [9, 12, 11, Decimal('0.0000397625')], [7, 12, 13, Decimal('0.0000397625')], [10, 13, 12, Decimal('0.0000397625')], [8, 13, 14, Decimal('0.0000397625')], [8, 13, 15, Decimal('0.0000397625')], [10, 15, 11, Decimal('0.0000397625')], [7, 15, 14, Decimal('0.0000397625')], [10, 13, 2, Decimal('0.000078125')], [0, 0, 2, Decimal('0.000078825')], [2, 1, 2, Decimal('0.000078825')], [6, 2, 0, Decimal('0.000078825')], [0, 6, 7, Decimal('0.000078825')], [3, 8, 9, Decimal('0.000078825')], [11, 11, 10, Decimal('0.000078825')], [11, 11, 12, Decimal('0.000078825')], [5, 4, 5, Decimal('0.0000795250')], [5, 4, 6, Decimal('0.0000795250')], [6, 12, 13, Decimal('0.0000795250')], [6, 15, 14, Decimal('0.0000795250')], [1, 8, 9, Decimal('0.0001178875')], [9, 11, 10, Decimal('0.0001178875')], [10, 2, 0, Decimal('0.000156950')], [2, 8, 9, Decimal('0.000156950')], [10, 11, 10, Decimal('0.000156950')], [9, 0, 4, Decimal('0.0005007')], [11, 0, 7, Decimal('0.0005007')], [11, 0, 8, Decimal('0.0005007')], [8, 1, 4, Decimal('0.0005007')], [7, 1, 5, Decimal('0.0005007')], [7, 1, 9, Decimal('0.0005007')], [4, 2, 10, Decimal('0.0005007')], [5, 2, 12, Decimal('0.0005007')], [5, 2, 13, Decimal('0.0005007')], [5, 2, 15, Decimal('0.0005007')], [3, 3, 11, Decimal('0.0005007')], [3, 3, 15, Decimal('0.0005007')], [11, 0, 4, Decimal('0.000578825')], [6, 0, 7, Decimal('0.000578825')], [6, 0, 8, Decimal('0.000578825')], [7, 1, 4, Decimal('0.000578825')], [8, 1, 5, Decimal('0.000578825')], [8, 1, 9, Decimal('0.000578825')], [5, 2, 10, Decimal('0.000578825')], [0, 2, 12, Decimal('0.000578825')], [0, 2, 13, Decimal('0.000578825')], [0, 2, 15, Decimal('0.000578825')], [1, 3, 11, Decimal('0.000578825')], [1, 3, 15, Decimal('0.000578825')], [9, 4, 5, Decimal('0.000579525')], [8, 4, 6, Decimal('0.000579525')], [8, 4, 8, Decimal('0.000579525')], [7, 5, 6, Decimal('0.000579525')], [11, 7, 6, Decimal('0.000579525')], [11, 8, 9, Decimal('0.000579525')], [7, 9, 8, Decimal('0.000579525')], [4, 10, 11, Decimal('0.000579525')], [3, 11, 10, Decimal('0.000579525')], [3, 11, 12, Decimal('0.000579525')], [5, 12, 11, Decimal('0.000579525')], [5, 13, 14, Decimal('0.000579525')], [3, 15, 13, Decimal('0.000579525')], [3, 15, 14, Decimal('0.000579525')], [9, 4, 6, Decimal('0.0006185875')], [9, 4, 8, Decimal('0.0006185875')], [11, 6, 5, Decimal('0.0006192875')], [7, 6, 7, Decimal('0.0006192875')], [8, 8, 7, Decimal('0.0006192875')], [4, 11, 12, Decimal('0.0006192875')], [4, 11, 15, Decimal('0.0006192875')], [6, 0, 4, Decimal('0.000656950')], [9, 0, 7, Decimal('0.000656950')], [10, 0, 8, Decimal('0.000656950')], [0, 2, 10, Decimal('0.000656950')], [2, 2, 12, Decimal('0.000656950')], [4, 2, 13, Decimal('0.000656950')], [2, 2, 15, Decimal('0.000656950')], [6, 7, 6, Decimal('0.000657650')], [6, 8, 9, Decimal('0.000657650')], [1, 11, 10, Decimal('0.000657650')], [0, 12, 11, Decimal('0.000657650')], [0, 13, 14, Decimal('0.000657650')], [1, 15, 13, Decimal('0.000657650')], [1, 15, 14, Decimal('0.000657650')], [1, 11, 12, Decimal('0.0006583500')], [9, 8, 9, Decimal('0.0006967125')], [4, 15, 14, Decimal('0.0006967125')], [6, 6, 5, Decimal('0.0006974125')], [10, 0, 4, Decimal('0.000735075')], [10, 0, 7, Decimal('0.000735075')], [2, 2, 10, Decimal('0.000735075')], [2, 2, 13, Decimal('0.000735075')], [10, 8, 9, Decimal('0.000735775')], [2, 12, 11, Decimal('0.000735775')], [2, 15, 14, Decimal('0.000735775')], [10, 4, 5, Decimal('0.000813900')], [10, 4, 6, Decimal('0.000813900')]]


    # --- 构造拓扑（里面会把 packet_size 真正按 num_chunk 均分）---
    topo = NVD2_1_topology(num_chunk=int(config.num_chunk), packet_size=config.packet_size)
    G = topo.topology

    # --- 提取 GPU 节点 ---
    gpu_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "GPU"]
    gpu_nodes.sort()

    # --- 边界路由器与跨DC对信息日志 ---
    _dcs, _dc_to_brs, _pairs = enumerate_br_pairs(G)
    log(f"Border-routers per DC: {_dc_to_brs}")
    log(f"Allowed BR pairs: {_pairs}")

    # --- 每个 subchunk 实际大小（与拓扑保持一致）---
    packet_size_per_subchunk = topo.packet_size  # 已是 Decimal

    log("GPU nodes:", gpu_nodes)
    log("Per-subchunk size:", packet_size_per_subchunk)

    # --- 先构造初始树：优先使用外部 policy 热启动（每条记录：[chunk_id, src, dst, latency]）---
    all_subs = []
    sub_src_of = {}
    num_chunks_per_gpu = 1
    num_subchunks_per_chunk = int(config.num_chunk)
    for src in gpu_nodes:
        for _c in range(num_chunks_per_gpu):
            for _k in range(num_subchunks_per_chunk):
                sid = len(all_subs)
                all_subs.append(sid)
                sub_src_of[sid] = src

    used_policy_hotstart = False
    try:
        if 'policy' in globals() and isinstance(policy, list) and len(policy) > 0:
            init_trees, init_depths, init_max_depths, init_choice = build_trees_from_policy(
                G, all_subs, sub_src_of, packet_size_per_subchunk, gpu_nodes, policy
            )
            used_policy_hotstart = True
            log("[Init] Hot-start from policy list succeeded.")
        else:
            raise RuntimeError("No valid policy provided.")
    except Exception as e:
        log(f"[Init] Policy hot-start failed ({e}), fallback to BFS-based initializer.")
        init_trees, init_depths, init_max_depths, init_choice = build_optimized_trees(
            G, all_subs, sub_src_of, packet_size_per_subchunk, gpu_nodes
        )

    # --- Evaluate raw policy as-is (directly build trees from policy and simulate) ---
    try:
        trees_raw, depths_raw, maxd_raw, choice_raw = build_trees_from_policy(
            G, all_subs, sub_src_of, packet_size_per_subchunk, gpu_nodes, policy
        )
        ms_raw, per_raw = simulate_allgather_pipeline_bfs(
            G, packet_size_per_subchunk, gpu_nodes,
            trees_override=trees_raw, depths_override=depths_raw, max_depths_override=maxd_raw,
            verbose=True
        )
        log(f"[Baseline] Raw policy makespan: {float(ms_raw)*1e6:.3f} us")
    except Exception as e:
        log(f"[Baseline] Raw policy evaluation failed: {e}")

    # --- Policy-only queue ordering baseline (uses init_trees as structure, orders children by policy times) ---
    ms_policy, _, trees_policy = _evaluate_policy_baseline(
        G, packet_size_per_subchunk, gpu_nodes,
        init_trees, init_depths, init_max_depths, sub_src_of, policy
    )
    log(f"[Baseline] Policy-queue-order makespan: {float(ms_policy)*1e6:.3f} us")

    # --- Tail-driven per-node priority refinement (only changes child order) ---
    ms_tail, trees_tail, _, policy_map_after = _bottleneck_reprioritize_policy(
            G, packet_size_per_subchunk, gpu_nodes,
            init_trees, init_depths, init_max_depths, sub_src_of, policy,
            rounds=60, alpha=Decimal('0.25')
        )
    log(f"[Baseline] Tail-driven priority makespan: {float(ms_tail)*1e6:.3f} us")
    if ms_tail + Decimal('1e-18') < ms_policy:
        # Use better ordering as new init (edges unchanged)
        init_trees = trees_tail

    # --- 先用 PBMS 在“组合层面”做全局搜索，再用 LNS+Beam 精修 ---
    log("[MAIN] Calling PBMS global search ...")
    best_trees, best_depths, best_maxd, best_choice, best_ms = optimize_tree_combo_pbm(
            G, packet_size_per_subchunk, gpu_nodes,
            all_subs, sub_src_of,
            init_trees, init_depths, init_max_depths, init_choice,
            pop_size=8, generations=15, elite_keep=3, mutation_per_iter=4
        )
    log(f"[MAIN] PBMS returned makespan: {float(best_ms)*1e6:.3f} us")
    log(f"\n[Optimizer] PBMS makespan: {float(best_ms):.9f}")

    # 可选：在 PBMS 基础上做 LNS+Beam 的局部精修
    log("[MAIN] Calling LNS critical-beam ...")
    best_trees, best_depths, best_maxd, best_choice, best_ms = lns_critical_beam(
            G, packet_size_per_subchunk, gpu_nodes,
            all_subs, sub_src_of,
            best_trees, best_depths, best_maxd, best_choice,
            beam_width=6, outer_iters=10, local_rewire_trials=50
        )
    log(f"[MAIN] LNS returned makespan: {float(best_ms)*1e6:.3f} us")
    log(f"[Optimizer] LNS+Beam best makespan: {float(best_ms):.9f}")

    # --- 使用 HHPR 进一步全局搜索（更激进） ---
    log("[MAIN] Calling HHPR global search ...")
    best_trees, best_depths, best_maxd, best_choice, best_ms = hhpr_global_search(
            G, packet_size_per_subchunk, gpu_nodes,
            all_subs, sub_src_of,
            best_trees, best_depths, best_maxd, best_choice,
            generations=12, iters_per_gen=40, elite_size=6,
            l2s_advisor=L2SAdvisor(),           # placeholder advisor; replace with your loaded GNN policy
            log_l2s_dataset=True,
            l2s_dataset_path="l2s_dataset.jsonl"
        )
    log(f"[MAIN] HHPR returned makespan: {float(best_ms)*1e6:.3f} us")
    log(f"[Optimizer] HHPR (Hyper-heuristic + PathRelinking) best makespan: {float(best_ms):.9f}")

    # --- Compare with policy and tail-refined baselines after optimization ---
    if ms_policy + Decimal('1e-18') < best_ms:
        log(f"[Select] Policy baseline beats search: {float(ms_policy):.3f} us vs {float(best_ms):.3f} us")
        best_ms = ms_policy
        best_trees = trees_policy
        best_depths = init_depths
        best_maxd = init_max_depths

    if ms_tail + Decimal('1e-18') < best_ms:
        log(f"[Select] Tail-refined baseline beats search: {float(ms_tail):.3f} us vs {float(best_ms):.3f} us")
        best_ms = ms_tail
        best_trees = trees_tail
        best_depths = init_depths
        best_maxd = init_max_depths

    # --- 使用最优树组合再跑一次详细带日志的流水线模拟 ---
    simulate_allgather_pipeline_bfs(
        G, packet_size_per_subchunk, gpu_nodes,
        trees_override=best_trees, depths_override=best_depths, max_depths_override=best_maxd,
        verbose=True
    )
    # --- 显式日志优化结果 ---
    log(f"[Final] Optimized makespan: {float(best_ms)*1e6:.3f} us")
    log("[Final] Optimization completed with PBMS + LNS + HHPR")
    t_end = time.time()
    log(f"[Runtime] Total execution time: {t_end - t_start:.3f} seconds")