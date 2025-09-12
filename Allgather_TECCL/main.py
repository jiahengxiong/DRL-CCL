import bisect
import sys
import time

import config
import networkx as nx
import matplotlib.pyplot as plt
from utils.NVD2_1_topology import NVD2_1_topology
from decimal import Decimal
from Allgather_TECCL.utils.tools import add_node_job, select_node_job, start_send, combine_job, end_send, start_receive, \
    end_receive, \
    check_buffer
from Allgather_TECCL.utils.util import load_topology


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


def build_broad_shallow_trees(graph, nodes):
    """
    Build a tree for each node as the root, ensuring breadth is maximized
    and depth is minimized.

    Parameters:
    - graph: NetworkX Graph representing the topology.
    - nodes: List of nodes in the graph.

    Returns:
    - dict: A dictionary where keys are root nodes and values are BFS trees (in NetworkX format).
    """
    BFS_trees = {}

    for root_node in nodes:
        # Perform BFS starting from the root node
        bfs_tree = nx.bfs_tree(graph, source=root_node)
        BFS_trees[root_node] = bfs_tree

    return BFS_trees


def broad_cast(tree, current_roots, visited, completion_times, packet_size):
    """
    Perform a single step of broadcast for the given tree with time simulation.

    Parameters:
    - tree: BFS tree (NetworkX DiGraph) rooted at the root node.
    - current_roots: List of nodes broadcasting in this step.
    - visited: Set of nodes that have already received the message.
    - completion_times: Dict tracking when each node finishes broadcasting.
    - packet_size: Size of the packet being sent (in GB).

    Returns:
    - next_roots: List of nodes that will broadcast in the next step.
    """
    next_roots = []
    for root in current_roots:
        root_time = completion_times[root]  # When the root finished its last broadcast
        for neighbor in tree.successors(root):
            if neighbor not in visited:
                # Get link properties
                edge_data = tree.get_edge_data(root, neighbor)
                bandwidth = edge_data["link_capcapacity"]  # GB/s
                propagation_latency = edge_data["propagation_latency"]  # seconds

                # Calculate transmission time and arrival time
                transmission_time = packet_size / bandwidth
                arrival_time = root_time + transmission_time + propagation_latency

                # Update completion time if earlier or unvisited
                if neighbor not in completion_times or arrival_time < completion_times[neighbor]:
                    completion_times[neighbor] = arrival_time

                visited.add(neighbor)
                next_roots.append(neighbor)
    return next_roots


def main(policy):
    # datacenter = NVD2_1_topology(packet_size=config.packet_size, num_chunk=config.num_chunk)  # Packet size in GB
    datacenter = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk, chassis=config.chassis,
                               name=config.topology_name)
    node_list = datacenter.topology.nodes()
    print(f'Number of nodes: {len(node_list)}')
    # gpu_list = datacenter.gpus
    NVD2_topology = datacenter.topology
    buffer_matrix = [[0 for _ in range(len(node_list) * config.num_chunk * config.buffer_constant)] for _ in
                     range(len(node_list) * config.num_chunk * config.buffer_constant)]
    sent_matrix = [[0 for _ in range(len(node_list) * config.num_chunk * config.buffer_constant)] for _ in
                   range(len(node_list) * config.num_chunk * config.buffer_constant)]
    print(len(node_list) * config.num_chunk)
    for node in node_list:
        memory = NVD2_topology.nodes[node]['memory']
        for buffer_index, buffer in memory.items():
            if buffer['buffer'] is not None:
                buffer_matrix[node][buffer_index] = 1

    # Generate the BFS trees
    time = Decimal(str(0.0))
    for P in policy:
        sender = P[1]
        # print(P)
        # print(sender, NVD2_topology.nodes[sender])
        NVD2_topology.nodes[sender]['policy'].append(P)

    # for time in decimal_range(0, 10, '0.0000000015625'):
    while check_buffer(topology=NVD2_topology) is False:
        # for time in range(1):
        # for time_step in range(5):
        # for time in range(2):

        """print("********job in each node before select********")
        for node in node_list:
            print(node, NVD2_topology.nodes[node]['job'])
        print("############################################")"""

        """for node in node_list:
            select_node_job(topology=NVD2_topology, dst=node)"""

        """for node in node_list:
            combine_job(topology=NVD2_topology, node=node)"""
        for link in NVD2_topology.edges:
            end_send(topology=NVD2_topology, link=link, time=time)
        for link in NVD2_topology.edges:
            start_receive(topology=NVD2_topology, link=link, time=time)
        for link in NVD2_topology.edges:
            end_receive(topology=NVD2_topology, link=link, time=time)

        for node in node_list:
            add_node_job(topology=NVD2_topology, src=node, time=time, memory_state=buffer_matrix,
                         sent_matrix=sent_matrix)

        for node in node_list:
            start_send(topology=NVD2_topology, node=node, time=time, memory_state=buffer_matrix)

        event_list = []

        for link in NVD2_topology.edges:
            job_list = NVD2_topology.edges[link]['job']
            for job in job_list:
                event_list.append(job['send_time'])
                event_list.append(job['sent_time'])
                event_list.append(job['receive_time'])
                event_list.append(job['received_time'])

        sorted_event_list = sorted(event_list)
        index = bisect.bisect_right(sorted_event_list, time)
        # print(time, sorted_event_list)
        if index < len(event_list):
            time = sorted_event_list[index]
        else:

            print('finish time:', time)
        """else:
            time = time + Decimal(str(0.0000000000001))"""

        """print("********job in each node after select********")
        for node in node_list:
            print(node, NVD2_topology.nodes[node]['job'])
        print("############################################")"""

        # print(buffer_matrix)

        """print(f"***********************{time}********************")
        for node in node_list:
            print(node, NVD2_topology.nodes[node]['memory'])"""

        if check_buffer(topology=NVD2_topology):
            print(
                f"-----topology: {config.topology_name}, packet_size: {config.packet_size} chassis: {config.chassis}-----")
            print(f"Finish time: {time * 1000000} us")
            break
        # print(time)
        """for node in NVD2_topology.nodes:
            state = NVD2_topology.nodes[node]"""
        # print(time)
        # config.connect_matrix = []
    # print(NVD2_topology.nodes())
    # trans_dict = {}
    # pro_dict = {}
    # for i in NVD2_topology.nodes():
    #     for j in NVD2_topology.nodes():
    #         if i != j:
    #             if NVD2_topology.has_edge(i, j):
    #                 print(i, j, float(NVD2_topology.edges[i, j]['transmission_latency']))
    #                 trans_dict[f'{i + 1}->{j + 1}'] = float(NVD2_topology.edges[i, j]['transmission_latency'])
    #                 pro_dict[f'{i + 1}->{j + 1}'] = float(NVD2_topology.edges[i, j]['propagation_latency'])
    # print(trans_dict)
    # print(pro_dict)
    for node in NVD2_topology.nodes:
        print(node, NVD2_topology.nodes[node]['memory'])


if __name__ == "__main__":
    policy = [[4, 4, 0, 0, 0], [4, 4, 5, 0, 0], [4, 4, 7, 0, 0], [4, 4, 8, 0, 0], [4, 4, 9, 0, 0], [5, 5, 4, 1, 0], [5, 5, 6, 1, 0], [5, 5, 7, 1, 0], [5, 5, 8, 1, 0], [5, 5, 9, 1, 0], [6, 6, 1, 2, 0], [6, 6, 5, 2, 0], [6, 6, 7, 2, 0], [6, 6, 8, 2, 0], [6, 6, 9, 2, 0], [7, 7, 4, 3, 0], [7, 7, 5, 3, 0], [7, 7, 6, 3, 0], [7, 7, 8, 3, 0], [7, 7, 9, 3, 0], [8, 8, 0, 4, 0], [8, 8, 4, 4, 0], [8, 8, 5, 4, 0], [8, 8, 6, 4, 0], [8, 8, 7, 4, 0], [8, 8, 9, 4, 0], [9, 9, 4, 5, 0], [9, 9, 5, 5, 0], [9, 9, 6, 5, 0], [9, 9, 7, 5, 0], [9, 9, 8, 5, 0], [10, 10, 11, 6, 0], [10, 10, 12, 6, 0], [10, 10, 13, 6, 0], [10, 10, 14, 6, 0], [11, 11, 10, 7, 0], [11, 11, 12, 7, 0], [11, 11, 13, 7, 0], [11, 11, 15, 7, 0], [12, 12, 3, 8, 0], [12, 12, 10, 8, 0], [12, 12, 11, 8, 0], [12, 12, 13, 8, 0], [12, 12, 14, 8, 0], [12, 12, 15, 8, 0], [13, 13, 2, 9, 0], [13, 13, 10, 9, 0], [13, 13, 11, 9, 0], [13, 13, 12, 9, 0], [13, 13, 14, 9, 0], [13, 13, 15, 9, 0], [14, 14, 3, 10, 0], [14, 14, 10, 10, 0], [14, 14, 12, 10, 0], [14, 14, 13, 10, 0], [14, 14, 15, 10, 0], [15, 15, 2, 11, 0], [15, 15, 11, 11, 0], [15, 15, 12, 11, 0], [15, 15, 13, 11, 0], [15, 15, 14, 11, 0], [4, 7, 6, 0, 2], [5, 6, 0, 1, 2], [5, 9, 1, 1, 2], [6, 7, 4, 2, 2], [9, 6, 1, 5, 2], [10, 11, 2, 6, 2], [10, 11, 15, 6, 2], [11, 10, 14, 7, 2], [11, 15, 3, 7, 2], [14, 10, 11, 10, 2], [15, 11, 10, 11, 2], [4, 0, 2, 0, 3], [6, 1, 2, 2, 3], [8, 0, 3, 4, 3], [12, 3, 0, 8, 3], [13, 2, 0, 9, 3], [14, 3, 1, 10, 3], [15, 2, 1, 11, 3], [7, 5, 0, 3, 4], [9, 6, 1, 5, 4], [5, 0, 2, 1, 5], [5, 1, 3, 1, 5], [9, 1, 2, 5, 5], [10, 2, 0, 6, 5], [11, 3, 1, 7, 5], [4, 2, 13, 0, 6], [6, 2, 12, 2, 6], [8, 3, 14, 4, 6], [8, 3, 15, 4, 6], [12, 0, 4, 8, 6], [12, 0, 8, 8, 6], [13, 0, 9, 9, 6], [14, 1, 8, 10, 6], [15, 1, 9, 11, 6], [7, 0, 2, 3, 7], [9, 1, 3, 5, 7], [12, 4, 7, 8, 7], [12, 4, 9, 8, 7], [13, 9, 5, 9, 7], [13, 9, 7, 9, 7], [15, 9, 4, 11, 7], [4, 13, 10, 0, 8], [4, 13, 11, 0, 8], [4, 13, 12, 0, 8], [4, 13, 15, 0, 8], [5, 2, 10, 1, 8], [5, 3, 14, 1, 8], [8, 14, 13, 4, 8], [8, 15, 12, 4, 8], [9, 2, 13, 5, 8], [10, 0, 4, 6, 8], [10, 0, 5, 6, 8], [10, 0, 6, 6, 8], [10, 0, 8, 6, 8], [10, 0, 9, 6, 8], [11, 1, 9, 7, 8], [14, 8, 5, 10, 8], [14, 8, 6, 10, 8], [14, 8, 7, 10, 8], [15, 9, 5, 11, 8], [15, 9, 6, 11, 8], [15, 9, 7, 11, 8], [15, 9, 8, 11, 8], [4, 13, 14, 0, 9], [5, 10, 11, 1, 9], [5, 10, 12, 1, 9], [5, 10, 13, 1, 9], [5, 14, 15, 1, 9], [6, 12, 10, 2, 9], [6, 12, 11, 2, 9], [6, 12, 13, 2, 9], [6, 12, 14, 2, 9], [6, 12, 15, 2, 9], [8, 14, 10, 4, 9], [8, 15, 11, 4, 9], [10, 8, 7, 6, 9], [11, 9, 4, 7, 9], [11, 9, 5, 7, 9], [11, 9, 6, 7, 9], [11, 9, 7, 7, 9], [11, 9, 8, 7, 9], [12, 8, 5, 8, 9], [12, 8, 6, 8, 9], [13, 5, 4, 9, 9], [13, 5, 6, 9, 9], [13, 5, 8, 9, 9], [14, 8, 4, 10, 9], [14, 8, 9, 10, 9], [7, 2, 10, 3, 10], [7, 2, 11, 3, 10], [7, 2, 12, 3, 10], [7, 2, 13, 3, 10], [7, 2, 14, 3, 10], [7, 2, 15, 3, 10], [9, 3, 10, 5, 10], [9, 3, 11, 5, 10], [9, 3, 12, 5, 10], [9, 3, 14, 5, 10], [9, 3, 15, 5, 10]]
    start_time = time.time()

    # 你的代码
    config.packet_size = Decimal(str(4.0)) / Decimal(str(1024.0))
    config.num_chunk = 1
    config.chassis = 2
    config.collective = 'ALLGATHER'
    config.topology_name = 'NVD2'
    config.connect_matrix = []
    epoch_time = config.packet_size / Decimal('50.0')
    config.connectivity = 0.9
    # for p in range(len(policy)):
    #     policy[p] = [policy[p][0], policy[p][1], policy[p][2], policy[p][3] * epoch_time]
    # print(policy)
    main(policy)

    # 记录结束时间
    end_time = time.time()

    # 计算并打印运行时间（单位：秒）
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time:.6f} 秒")
