from collections import deque, defaultdict
from decimal import Decimal
import networkx as nx

#############################################
# 1) 基础工具
#############################################
def get_gpu_nodes(G):
    return [n for n in G.nodes if G.nodes[n].get('type') == 'GPU']

def edge_delay(G, u, v):
    e = G[u][v]
    # 统一转 Decimal，避免浮点误差
    return Decimal(str(e['propagation_latency'])) + Decimal(str(e['transmission_latency']))

def is_cross_dc(G, u, v):
    return G.nodes[u].get('DC') != G.nodes[v].get('DC')

def make_edge_key(u, v, bidirectional_as_one=True):
    return frozenset({u, v}) if bidirectional_as_one else (u, v)

#############################################
# 2) 受跨DC唯一约束的 BFS 构树
#############################################
def build_bfs_tree_with_cross_dc_limit(G, root, used_cross_edges, bidirectional_as_one=True):
    """
    为 root 构造一棵 BFS 树，同时遵守：一旦选择某条跨DC边作为树边，全局仅能使用一次。
    返回: parent(dict), levels(list[list])
    注：若跨DC资源不足导致某些节点不可达，树将只覆盖可达部分，这是物理约束下的合理结果。
    """
    parent = {root: None}
    q = deque([root])

    while q:
        u = q.popleft()
        for v in G.successors(u):
            if v in parent:
                continue
            # 若是跨 DC 边并已被别的树占用，跳过
            if is_cross_dc(G, u, v):
                k = make_edge_key(u, v, bidirectional_as_one)
                if k in used_cross_edges:
                    continue
            # 选择该边进树
            parent[v] = u
            if is_cross_dc(G, u, v):
                used_cross_edges.add(make_edge_key(u, v, bidirectional_as_one))
            q.append(v)

    # 生成按层次的节点列表（levels[0] = [root]）
    children = defaultdict(list)
    for v, u in parent.items():
        if u is not None:
            children[u].append(v)
    levels = [[root]]
    cur = [root]
    while True:
        nxt = []
        for x in cur:
            nxt.extend(children[x])
        if not nxt:
            break
        levels.append(nxt)
        cur = nxt

    return parent, levels

#############################################
# 3) 沿树转发一个 subchunk 的事件生成（无排队）
#############################################
def forward_one_subchunk(G, root, subchunk_id, parent, levels, t0=Decimal('0')):
    """
    沿 BFS 树做一次“洪泛式”传播：
    - 每个子节点在其父节点“到达时刻”立刻触发发送（不建队、不串行化端口）。
    - 链路时延 = e['propagation_latency'] + e['transmission_latency']。
    返回：
      schedule: [(send_time, u, v, subchunk_id, arrive_time), ...]
      arrive_time_at: {node: time}
    """
    arrive_time_at = {root: Decimal(t0)}
    schedule = []

    for lvl in levels[1:]:
        for v in lvl:
            u = parent.get(v)
            if u is None or (u not in arrive_time_at):
                continue
            send_time = arrive_time_at[u]
            d = edge_delay(G, u, v)
            t_arr = send_time + d
            arrive_time_at[v] = t_arr
            schedule.append((send_time, u, v, subchunk_id, t_arr))

    return schedule, arrive_time_at

#############################################
# 4) root 的 subchunk 编号规则（与你给的 initial_buffer 对齐）
#############################################
def subchunks_of_root(root, num_chunk, switch_num=4, buffer_constant=1):
    """
    ALLGATHER 场景下，你给的 initial_buffer 为：
    base = (root - switch_num) * num_chunk * buffer_constant
    这里完全沿用这一编号方式，使得每个 root 拥有一段连续 subchunk。
    """
    base = (root - switch_num) * num_chunk * buffer_constant
    return list(range(base, base + num_chunk * buffer_constant))

#############################################
# 5) 主流程：为所有 root 建树+转发
#############################################
def simulate_allgather_bfs_cross_limited(
    G,
    num_chunk,
    buffer_constant=1,
    start_time=Decimal('0'),
    bidirectional_as_one=True
):
    """
    - 所有 root 共享 used_cross_edges，保证任一跨DC物理边只被所有树使用一次。
    - 对每个 root 的每个 subchunk，沿其 BFS 树进行无排队的转发。
    返回：
      trees: {root: (parent, levels)}
      schedule_all: 全部链路事件，已按 (send_time, u, v) 排序
      arrive_time: {(subchunk_id, node): t_arrive}
      summary: dict，包含 makespan、不可达计数等
    """
    used_cross_edges = set()
    trees = {}
    schedule_all = []
    arrive_time = {}

    roots = get_gpu_nodes(G)

    # 先为所有 root 建树（共享跨DC资源约束）
    for r in roots:
        parent, levels = build_bfs_tree_with_cross_dc_limit(
            G, r, used_cross_edges,
            bidirectional_as_one=bidirectional_as_one
        )
        trees[r] = (parent, levels)

    # 再对每个 root 拥有的 subchunks 沿各自树转发
    for r in roots:
        parent, levels = trees[r]
        sc_list = subchunks_of_root(r, num_chunk, switch_num=4, buffer_constant=buffer_constant)
        for sc in sc_list:
            sch, arr = forward_one_subchunk(G, r, sc, parent, levels, t0=Decimal(start_time))
            schedule_all.extend(sch)
            for node, t in arr.items():
                arrive_time[(sc, node)] = t

    schedule_all.sort(key=lambda x: (x[0], x[1], x[2]))

    # 统计摘要
    all_arrival_times = [t for (_, _, _, _, t) in schedule_all]
    makespan = max(all_arrival_times) if all_arrival_times else Decimal(start_time)

    # 统计不可达（对每个 subchunk 期望到达所有 GPU；未出现则视为不可达）
    gpu_nodes = roots
    total_expected = len(gpu_nodes) * len(subchunks_of_root(roots[0], num_chunk, 4, buffer_constant))
    # 上面只是每个 root 的 subchunk 数。所有 root 的 subchunk 总数：
    per_root = len(subchunks_of_root(roots[0], num_chunk, 4, buffer_constant))
    total_expected = len(gpu_nodes) * per_root * len(gpu_nodes)

    delivered = 0
    for r in roots:
        for sc in subchunks_of_root(r, num_chunk, 4, buffer_constant):
            for n in gpu_nodes:
                if (sc, n) in arrive_time:
                    delivered += 1
    unreachable = total_expected - delivered

    summary = {
        "num_roots": len(roots),
        "num_subchunks_per_root": per_root,
        "events": len(schedule_all),
        "makespan": makespan,
        "unreachable_pairs": unreachable,
    }

    return trees, schedule_all, arrive_time, summary