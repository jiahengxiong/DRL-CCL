import heapq
import sys
import config
import networkx as nx
import matplotlib.pyplot as plt
from utils.NVD2_1_topology import NVD2_1_topology
from decimal import Decimal
from AlltoAll.utils.tools import add_node_job, start_send, end_send, start_receive, \
    end_receive, \
    check_buffer, Initialize_buffers
from AlltoAll.utils.util import load_topology
from collections import deque, defaultdict


def decimal_range(start, stop, step):
    """
    使用 Decimal 模块实现高精度浮点数遍历。
    - start: 起始值（Decimal 类型）
    - stop: 终止值（Decimal 类型）
    - step: 步长（Decimal 类型）
    """
    current = Decimal(start)
    stop = Decimal(stop)
    step = Decimal(step)

    while current < stop:
        yield float(current)  # 将 Decimal 转换为 float 输出
        current += step


def draw_tree_with_total_time(tree):
    # 使用 spring 布局计算节点位置
    pos = nx.spring_layout(tree)

    # 绘制节点、边和节点标签
    nx.draw_networkx_nodes(tree, pos, node_color='lightblue', node_size=200)
    nx.draw_networkx_edges(tree, pos, edge_color='gray')
    nx.draw_networkx_labels(tree, pos, font_size=10)

    # 提取每条边上的 total_time 属性，如果不存在则显示空字符串
    edge_labels = {(u, v): d.get("total_time", "") for u, v, d in tree.edges(data=True)}
    # 在边上绘制标签
    nx.draw_networkx_edge_labels(tree, pos, edge_labels=edge_labels, font_color='red', font_size=9)

    plt.title("Tree Graph with total_time on edges")
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()


def build_min_depth_far_node_first_tree(G, root):
    """
    在有向图 G 中，从 root 出发构造一棵“最小深度”树，同时在同一层里
    让累加传输时间更大的节点先进行扩展（先发远节点）。

    返回:
      T: nx.DiGraph, 构造好的树
      depth: dict[v] = v 的最小深度
      total_time: dict[v] = 不含排队的累加传输时间 (从 root 到 v 的路径边之和)
      parent: dict[v] = v 的父节点
    """

    # 1) 把所有边的 total_time 转成 Decimal
    for (u, v) in G.edges():
        G[u][v]['total_time'] = Decimal(str(G[u][v]['total_time']))

    nodes = list(G.nodes())
    depth = {}
    total_time = {}
    parent = {}

    # 初始化
    for v in nodes:
        depth[v] = float('inf')
        total_time[v] = Decimal("Infinity")
        parent[v] = None

    depth[root] = 0
    total_time[root] = Decimal("0.0")

    # 我们维护一个队列，但在同一层中，对节点按照 total_time[v] 降序进行扩展
    # data_in_queue = [(v, depth[v], total_time[v])]
    from heapq import heappush, heappop

    # Python heapq 是小根堆，所以我们用 depth[v] 为第一关键字(越小越优先)，
    # total_time[v] * -1 为第二关键字(越大越优先)
    # push/pop 的元素格式: (depth[v], -total_time[v], v)
    pq = []
    heappush(pq, (0, -total_time[root], root))

    while pq:
        d_u, neg_tu, u = heappop(pq)
        # 如果这个记录已经过期(即 depth[u] != d_u)，跳过
        if depth[u] != d_u:
            continue

        for v in G.successors(u):
            edge_t = G[u][v]['total_time']
            cand_depth = d_u + 1
            cand_time = total_time[u] + edge_t

            # 如果 v 还没访问过
            if depth[v] == float('inf'):
                # 直接更新
                depth[v] = cand_depth
                total_time[v] = cand_time
                parent[v] = u
                # 加入优先队列
                heappush(pq, (cand_depth, -cand_time, v))

            else:
                # 已访问过 => 比较
                if cand_depth < depth[v]:
                    # 更浅 => 一定更新
                    depth[v] = cand_depth
                    total_time[v] = cand_time
                    parent[v] = u
                    heappush(pq, (cand_depth, -cand_time, v))

                elif cand_depth == depth[v]:
                    # 同深度 => 如果 cand_time 更小，则更新
                    if cand_time < total_time[v]:
                        depth[v] = cand_depth
                        total_time[v] = cand_time
                        parent[v] = u
                        heappush(pq, (cand_depth, -cand_time, v))
                    # 也可以加上更多分散子节点的逻辑

    # 2) 构造树 T
    T = nx.DiGraph()
    for v in nodes:
        if v == root:
            continue
        p = parent[v]
        if p is not None:
            T.add_edge(p, v, total_time=G[p][v]['total_time'])

    return T


def build_tree(node_list, datacenter, cost_increase=Decimal("0.00")):
    """
    在 datacenter.topology 中，对每个非交换机节点都调用
    build_min_depth_far_node_first_tree() 构造树。
    同时，为了负载均衡，每构造完一棵树，就在图副本中
    对该树使用过的边 (u->v) 的 total_time 增加 cost_increase。

    参数：
      node_list: 可以是 G.nodes 或者你想遍历的节点集合
      datacenter: 拥有 topology 的对象 (datacenter.topology = G)
      cost_increase: 每次使用边后的负载增量 (默认为 0.1)

    返回：
      min_tree: dict, key=节点, value=构造出的树 (nx.DiGraph)
    """
    # 取出原图
    G = datacenter.topology

    # 复制原图，避免直接改动 datacenter.topology
    G_mod = nx.DiGraph()
    for (u, v, data) in G.edges(data=True):
        # 复制每条边的 total_time
        G_mod.add_edge(u, v, total_time=Decimal(str(data['total_time'])))

    min_tree = {}
    node_list = list(G.nodes())

    # 遍历节点
    for node in node_list:
        # 跳过 switch 节点
        if G.nodes[node]['type'] == "switch":
            continue

        # 调用你的单根树构造算法 (在图 G_mod 上)
        T = build_min_depth_far_node_first_tree(G_mod, node)
        # 保存
        min_tree[node] = T

        # 对该树使用过的边增加负载
        for (u, v) in T.edges():
            old_time = G_mod[u][v]['total_time']
            G_mod[u][v]['total_time'] = old_time + cost_increase

    return min_tree


def find_paths(trees, datacenter):
    path_dic = {}
    G = datacenter.topology
    node_list = list(G.nodes)
    for node in node_list:
        if G.nodes[node]['type'] != "switch":
            memory = G.nodes[node]['memory']
            tree = trees[node]
            for key, value in memory.items():
                buffer = value['buffer']
                dest_node = value['dest_node']
                if buffer is not None and dest_node is not node:
                    shortest_path = nx.shortest_path(tree, source=node, target=dest_node, weight="total_time",
                                                     method="dijkstra")
                    shortest_weight = nx.shortest_path_length(tree, source=node, target=dest_node, weight="total_time",
                                                              method="dijkstra")
                    path_dic[buffer] = {'path': shortest_path, 'total_time': shortest_weight}
    return path_dic


from collections import Counter


def count_edge_occurrences(trees_dict):
    edge_counter = Counter()

    for key, tree in trees_dict.items():
        if not isinstance(tree, nx.DiGraph):
            print(f"Warning: node_tree[{key}] is not a networkx.DiGraph, it is {type(tree)}")
            continue  # 跳过非图的对象

        edges = list(tree.edges())  # 转成列表，避免非迭代问题
        print(f"Processing tree {key}, edges: {edges}")  # 打印调试信息

        edge_counter.update(edges)

    return edge_counter


def main():
    # datacenter = NVD2_1_topology(packet_size=config.packet_size, num_chunk=config.num_chunk)  # Packet size in GB
    datacenter = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk, chassis=config.chassis,
                               name=config.topology_name)
    if config.topology_name == 'NVD2' and config.chassis == 1:
        # 获取所有名称为 'switch' 的节点（如果节点 id 就是 'switch'）
        nodes_to_remove = [n for n in datacenter.topology.nodes() if n == 'switch']
        datacenter.topology.remove_nodes_from(nodes_to_remove)
    # gpu_list = datacenter.gpus
    G = datacenter.topology
    node_list = list(G.nodes)
    node_tree = build_tree(node_list, datacenter)
    draw_tree_with_total_time(node_tree[0])
    path_dic = find_paths(node_tree, datacenter)
    print(path_dic)
    # edge_counts = count_edge_occurrences(node_tree)
    Initialize_buffers(datacenter.topology, path_dic)

    for time in decimal_range(0, 10, '5E-8'):
        # for time in range(10):
        # for time_step in range(5):
        time = Decimal(str(time))

        for link in G.edges:
            end_send(topology=G, link=link, time=time)
        """for link in G.edges:
            start_receive(topology=G, link=link, time=time)"""
        for link in G.edges:
            end_receive(topology=G, link=link, time=time)

        for node in node_list:
            add_node_job(topology=G, src=node, time=time)

        for node in node_list:
            start_send(topology=G, node=node, time=time)

        if check_buffer(topology=G):
            print(
                f"-----topology: {config.topology_name}, packet_size: {config.packet_size} chassis: {config.chassis}-----")
            print(f"Finish time: {time * 1000000} us")
            break


if __name__ == "__main__":
    config.packet_size = Decimal(str(200))
    config.num_chunk = 4
    config.chassis = 1
    config.collective = 'ALLTOALL'
    config.topology_name = 'NVD2'
    config.connect_matrix = []
    main()
