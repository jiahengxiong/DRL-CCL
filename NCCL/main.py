import bisect
import time

import config
import networkx as nx
from decimal import Decimal
from Shortest_path.utils.tools import add_node_job, start_send, end_send, start_receive, \
    end_receive, \
    check_buffer
from Shortest_path.utils.util import load_topology


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


def main(collective_time, policy):
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

    # # Generate the BFS trees
    # time = Decimal(str(0.0))
    # for P in policy:
    #     sender = P[1]
    #     # print(P)
    #     # print(sender, NVD2_topology.nodes[sender])
    #     NVD2_topology.nodes[sender]['policy'].append(P)
    DC_0_switch = [0, 1]
    DC_1_switch = [1, 0]
    DC_0_GPU = list(range(4, 4 + int((len(node_list)-4)/2)))
    DC_1_GPU = list(range(4 + int((len(node_list)-4)/2), len(node_list)))
    print(f'DC_0_switch: {DC_0_switch}')
    print(f'DC_1_switch: {DC_1_switch}')
    print(f'DC_0_GPU: {DC_0_GPU}')
    print(f'DC_1_GPU: {DC_1_GPU}')
    route_table = {}
    for node in node_list:
        memory = NVD2_topology.nodes[node]['memory']
        for buffer_index, buffer in memory.items():
            if buffer['buffer'] is not None:
                NVD2_topology.nodes[node]['receive_buffer'].append(buffer)
        print(node, NVD2_topology.nodes[node]['receive_buffer'])
    for node in node_list:
        # print(NVD2_topology.nodes[node]['memory'])
        for key, buffer in NVD2_topology.nodes[node]['memory'].items():
            if buffer['buffer'] is not None:
                target_list = DC_0_GPU + DC_1_GPU
                target_list.remove(node)
                route_table[buffer['buffer']] = {'target_list': target_list, 'policy': [], 'current_node': 0, 'c': node}
    # print(f'route_table: {route_table}')


    # for time in decimal_range(0, 10, '0.0000000015625'):
    Ring_topology = nx.DiGraph()
    Ring_topology.add_edge(0, 4, weight=1)
    Ring_topology.add_edge(4, 5, weight=1)
    Ring_topology.add_edge(5, 6, weight=1)
    Ring_topology.add_edge(6, 7, weight=1)
    Ring_topology.add_edge(7, 8, weight=1)
    Ring_topology.add_edge(8, 9, weight=1)
    Ring_topology.add_edge(9, 1, weight=1)
    Ring_topology.add_edge(1, 2, weight=1)
    Ring_topology.add_edge(2, 10, weight=1)
    Ring_topology.add_edge(10, 11, weight=1)
    Ring_topology.add_edge(11, 12, weight=1)
    Ring_topology.add_edge(12, 13, weight=1)
    Ring_topology.add_edge(13, 14, weight=1)
    Ring_topology.add_edge(14, 15, weight=1)
    Ring_topology.add_edge(15, 3, weight=1)
    Ring_topology.add_edge(3, 0, weight=1)
    for buffer_index, value in route_table.items():
        p = []
        target_list = value['target_list'].copy()  # 拷贝一份，避免污染
        current_node = value['c']

        while len(target_list) > 0:
            min_cost = float('inf')
            best_target = None
            best_path = []

            for target_node in target_list:
                path = nx.dijkstra_path(Ring_topology, source=current_node, target=target_node, weight='weight')
                cost = nx.dijkstra_path_length(Ring_topology, source=current_node, target=target_node,
                                               weight='weight')
                if buffer_index % config.num_chunk == 0:
                    for i in range(len(path) - 1):
                        NVD2_topology.edges[path[i], path[i+1]]['weight'] = NVD2_topology.edges[path[i], path[i+1]]['weight'] + 1
                if cost < min_cost:
                    min_cost = cost
                    best_target = target_node
                    best_path = path

            # 拼接整条路径（不去重）
            if len(p) > 0 and best_path[0] == p[-1]:
                p += best_path[1:]  # 避免重复当前点
            else:

                p += best_path

            # 更新状态
            current_node = best_target
            target_list.remove(best_target)

        # 更新策略路径
        route_table[buffer_index]['policy'] =   p

    # 打印查看结果
    for key, value in route_table.items():
        print(f'{key}: {value}')

    time = Decimal(str(0.0))
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
            start_receive(topology=NVD2_topology, link=link, time=time, route_table=route_table)
        for link in NVD2_topology.edges:
            end_receive(topology=NVD2_topology, link=link, time=time)

        for node in node_list:
            add_node_job(topology=NVD2_topology, src=node, time=time, memory_state=buffer_matrix,
                         sent_matrix=sent_matrix,route_table=route_table)

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
        # else:
        #
        #     print('finish time:', time)
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
            time_in_us = Decimal(str(time)) * Decimal('1000000')
            normalized_time = time_in_us.normalize()
            collective_time[connectivity][num_chunk][chunk] = f'{normalized_time} us'
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
    # for node in NVD2_topology.nodes:
    #     print(node, NVD2_topology.nodes[node]['memory'])


if __name__ == "__main__":
    policy = [[4, 4, 0, 0, 0], [4, 4, 5, 0, 0], [4, 4, 7, 0, 0], [4, 4, 8, 0, 0], [4, 4, 9, 0, 0], [5, 5, 4, 1, 0], [5, 5, 6, 1, 0], [5, 5, 7, 1, 0], [5, 5, 8, 1, 0], [5, 5, 9, 1, 0], [6, 6, 1, 2, 0], [6, 6, 5, 2, 0], [6, 6, 7, 2, 0], [6, 6, 8, 2, 0], [6, 6, 9, 2, 0], [7, 7, 4, 3, 0], [7, 7, 5, 3, 0], [7, 7, 6, 3, 0], [7, 7, 8, 3, 0], [7, 7, 9, 3, 0], [8, 8, 0, 4, 0], [8, 8, 4, 4, 0], [8, 8, 5, 4, 0], [8, 8, 6, 4, 0], [8, 8, 7, 4, 0], [8, 8, 9, 4, 0], [9, 9, 4, 5, 0], [9, 9, 5, 5, 0], [9, 9, 6, 5, 0], [9, 9, 7, 5, 0], [9, 9, 8, 5, 0], [10, 10, 11, 6, 0], [10, 10, 12, 6, 0], [10, 10, 13, 6, 0], [10, 10, 14, 6, 0], [11, 11, 10, 7, 0], [11, 11, 12, 7, 0], [11, 11, 13, 7, 0], [11, 11, 15, 7, 0], [12, 12, 3, 8, 0], [12, 12, 10, 8, 0], [12, 12, 11, 8, 0], [12, 12, 13, 8, 0], [12, 12, 14, 8, 0], [12, 12, 15, 8, 0], [13, 13, 2, 9, 0], [13, 13, 10, 9, 0], [13, 13, 11, 9, 0], [13, 13, 12, 9, 0], [13, 13, 14, 9, 0], [13, 13, 15, 9, 0], [14, 14, 3, 10, 0], [14, 14, 10, 10, 0], [14, 14, 12, 10, 0], [14, 14, 13, 10, 0], [14, 14, 15, 10, 0], [15, 15, 2, 11, 0], [15, 15, 11, 11, 0], [15, 15, 12, 11, 0], [15, 15, 13, 11, 0], [15, 15, 14, 11, 0], [4, 7, 6, 0, 2], [5, 6, 0, 1, 2], [5, 9, 1, 1, 2], [6, 7, 4, 2, 2], [9, 6, 1, 5, 2], [10, 11, 2, 6, 2], [10, 11, 15, 6, 2], [11, 10, 14, 7, 2], [11, 15, 3, 7, 2], [14, 10, 11, 10, 2], [15, 11, 10, 11, 2], [4, 0, 2, 0, 3], [6, 1, 2, 2, 3], [8, 0, 3, 4, 3], [12, 3, 0, 8, 3], [13, 2, 0, 9, 3], [14, 3, 1, 10, 3], [15, 2, 1, 11, 3], [7, 5, 0, 3, 4], [9, 6, 1, 5, 4], [5, 0, 2, 1, 5], [5, 1, 3, 1, 5], [9, 1, 2, 5, 5], [10, 2, 0, 6, 5], [11, 3, 1, 7, 5], [4, 2, 13, 0, 7], [6, 2, 12, 2, 7], [7, 0, 2, 3, 7], [8, 3, 14, 4, 7], [8, 3, 15, 4, 7], [9, 1, 3, 5, 7], [12, 0, 4, 8, 7], [12, 0, 8, 8, 7], [13, 0, 9, 9, 7], [14, 1, 8, 10, 7], [15, 1, 9, 11, 7], [12, 4, 7, 8, 8], [12, 4, 9, 8, 8], [13, 9, 5, 9, 8], [13, 9, 7, 9, 8], [15, 9, 4, 11, 8], [4, 13, 10, 0, 9], [4, 13, 11, 0, 9], [4, 13, 12, 0, 9], [4, 13, 15, 0, 9], [5, 2, 10, 1, 9], [5, 3, 14, 1, 9], [8, 14, 13, 4, 9], [8, 15, 12, 4, 9], [9, 2, 13, 5, 9], [10, 0, 4, 6, 9], [10, 0, 5, 6, 9], [10, 0, 6, 6, 9], [10, 0, 8, 6, 9], [10, 0, 9, 6, 9], [11, 1, 9, 7, 9], [14, 8, 5, 10, 9], [14, 8, 6, 10, 9], [14, 8, 7, 10, 9], [15, 9, 5, 11, 9], [15, 9, 6, 11, 9], [15, 9, 7, 11, 9], [15, 9, 8, 11, 9], [4, 13, 14, 0, 10], [5, 10, 11, 1, 10], [5, 10, 12, 1, 10], [5, 10, 13, 1, 10], [5, 14, 15, 1, 10], [6, 12, 10, 2, 10], [6, 12, 11, 2, 10], [6, 12, 13, 2, 10], [6, 12, 14, 2, 10], [6, 12, 15, 2, 10], [8, 14, 10, 4, 10], [8, 15, 11, 4, 10], [10, 8, 7, 6, 10], [11, 9, 4, 7, 10], [11, 9, 5, 7, 10], [11, 9, 6, 7, 10], [11, 9, 7, 7, 10], [11, 9, 8, 7, 10], [12, 8, 5, 8, 10], [12, 8, 6, 8, 10], [13, 5, 4, 9, 10], [13, 5, 6, 9, 10], [13, 5, 8, 9, 10], [14, 8, 4, 10, 10], [14, 8, 9, 10, 10], [7, 2, 10, 3, 11], [7, 2, 11, 3, 11], [7, 2, 12, 3, 11], [7, 2, 13, 3, 11], [7, 2, 14, 3, 11], [7, 2, 15, 3, 11], [9, 3, 10, 5, 11], [9, 3, 11, 5, 11], [9, 3, 12, 5, 11], [9, 3, 14, 5, 11], [9, 3, 15, 5, 11]]
    num_chunk_list = [1,2,4]
    chunk_size_list = [1, 4, 16, 64, 256]
    connectivity_list = [0.5]
    collective_time = {}
    execute_time = {}

    for connectivity in connectivity_list:
        collective_time[connectivity] = {}
        execute_time[connectivity] = {}
        for num_chunk in num_chunk_list:
            collective_time[connectivity][num_chunk] = {}
            execute_time[connectivity][num_chunk] = {}
            for chunk in chunk_size_list:
                collective_time[connectivity][num_chunk][chunk] = 0
                execute_time[connectivity][num_chunk][chunk] = 0
    for connectivity in connectivity_list:
        for num_chunk in num_chunk_list:
            for chunk in chunk_size_list:
                start_time = time.time()

                # 你的代码
                config.packet_size = Decimal(str(chunk)) / Decimal(str(1024.0))
                config.num_chunk = num_chunk
                config.connectivity = connectivity
                config.chassis = 2
                config.collective = 'ALLGATHER'
                config.topology_name = 'NVD2'
                config.connect_matrix = []
                main(collective_time, policy)

                # 记录结束时间
                end_time = time.time()

                # 计算并打印运行时间（单位：秒）
                execution_time = end_time - start_time
                execute_time[connectivity][num_chunk][chunk] = execution_time
                print(f"代码运行时间: {execution_time:.6f} 秒")
    print('collective_time:', collective_time)
    print('execute_time:', execute_time)
