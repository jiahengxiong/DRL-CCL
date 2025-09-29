import bisect
import sys
import time

import config
import networkx as nx
import matplotlib.pyplot as plt


from utils.NVD2_1_topology import NVD2_1_topology
from decimal import Decimal
from Allgather.utils.tools import add_node_job, select_node_job, start_send, combine_job, end_send, start_receive, end_receive, \
    check_buffer,queue
from Allgather.utils.util import load_topology


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
    datacenter = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk, chassis=config.chassis, name=config.topology_name)
    node_list = datacenter.topology.nodes()
    print(f'Number of nodes: {len(node_list)}')
    # gpu_list = datacenter.gpus
    NVD2_topology = datacenter.topology
    buffer_matrix = [[0 for _ in range(len(node_list) * config.num_chunk * config.buffer_constant)] for _ in
                     range(len(node_list) * config.num_chunk * config.buffer_constant)]
    sent_matrix = [[0 for _ in range(len(node_list) * config.num_chunk * config.buffer_constant)] for _ in
                   range(len(node_list) * config.num_chunk * config.buffer_constant)]
    DC_0_buffer = []
    DC_1_buffer = []

    print(len(node_list) * config.num_chunk)
    for node in node_list:
        memory = NVD2_topology.nodes[node]['memory']
        for buffer_index, buffer in memory.items():
            if buffer['buffer'] is not None:
                buffer_matrix[node][buffer_index] = 1
                if NVD2_topology.nodes[node]['DC'] == 0:
                    DC_0_buffer.append(buffer['buffer'])
                else:
                    DC_1_buffer.append(buffer['buffer'])
    print("DC_0_buffer", DC_0_buffer)
    print("DC_1_buffer", DC_1_buffer)


    # Generate the BFS trees
    time = Decimal(str(0.0))
    WAN_buffer = []

    #for time in decimal_range(0, 10, '0.0000000015625'):
    while check_buffer(topology=NVD2_topology) is False:
        # for time in range(1):
        # for time_step in range(5):
        # for time in range(2):


        # GPU_list = []
        # for n in NVD2_topology.nodes:
        #     if NVD2_topology.nodes[n]['type'] == 'GPU':
        #         GPU_list.append(n)
        buffer_num_dict = {}
        # for GPU in GPU_list:
        #     gpu_memory = NVD2_topology.nodes[GPU]['memory']
        #     for buffer, value in gpu_memory.items():
        #         # print(buffer, value)
        #         if value['buffer'] is not None:
        #             if buffer in buffer_num_dict.keys():
        #                 buffer_num_dict[buffer] += 1
        #             else:
        #                 buffer_num_dict[buffer] = 1
        # print(buffer_num_dict)

        """print("********job in each node before select********")
        for node in node_list:
            print(node, NVD2_topology.nodes[node]['job'])
        print("############################################")"""

        """for node in node_list:
            select_node_job(topology=NVD2_topology, dst=node)"""

        """for node in node_list:
            combine_job(topology=NVD2_topology, node=node)"""
        for link in NVD2_topology.edges:
            end_send(topology=NVD2_topology, link=link, time=time, WAN_buffer=WAN_buffer)
        for link in NVD2_topology.edges:
            start_receive(topology=NVD2_topology, link=link, time=time)
        for link in NVD2_topology.edges:
            end_receive(topology=NVD2_topology, link=link, time=time)

        for node in node_list:
            add_node_job(topology=NVD2_topology, src=node, time=time, memory_state=buffer_matrix,
                         sent_matrix=sent_matrix, DC0=DC_0_buffer,DC1=DC_1_buffer, WAN_buffer=WAN_buffer, buffer_num_dict=buffer_num_dict)

        queue(topology=NVD2_topology,memory_state=buffer_matrix,)

        for node in node_list:
            start_send(topology=NVD2_topology, node=node, time=time, memory_state=buffer_matrix, WAN_buffer=WAN_buffer,DC0=DC_0_buffer,DC1=DC_1_buffer, policy=policy)


        event_list = []
        # 使用 set 去重，效率更高
        event_set = set(event_list)

        for link in NVD2_topology.edges:
            job_list = NVD2_topology.edges[link]['job']
            job_max = []
            removed_jobs = []
            time_list = []
            for job in job_list:
                for key in ['send_time', 'sent_time', 'receive_time', 'received_time']:
                    t = job[key]
                    time_list.append(t)
                    job_max.append(t)
                    if t > time:
                        event_set.add(t)
                if max(job_max) < time - (max(time_list) - min(time_list)):
                    removed_jobs.append(job)
            for job in removed_jobs:
                NVD2_topology.edges[link]['job'].remove(job)


        # 如果你还需要保持为列表
        event_list = list(event_set)

        # for link in NVD2_topology.edges:
        #     job_list = NVD2_topology.edges[link]['job']
        #     for job in job_list:
        #         event_list.append(job['send_time'])
        #         event_list.append(job['sent_time'])
        #         event_list.append(job['receive_time'])
        #         event_list.append(job['received_time'])


        sorted_event_list = sorted(event_list)
        index = bisect.bisect_right(sorted_event_list, time)
        # print(time, sorted_event_list)
        if index < len(event_list):
            time = sorted_event_list[index]
        # else:
        #     time = time + Decimal("1e-12")
        # else:
        #     print(f"Finish time: {time * 1000000} us")
        #     print("--------------------------------")
        #     for node in node_list:
        #         memory = NVD2_topology.nodes[node]['memory']
        #         print(node, memory)
        #     break
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
        # print("job at 11", NVD2_topology.nodes[11]['job'][(11, 10)])
        # print("job at 2", NVD2_topology.nodes[2]['job'][(2, 10)])

        if check_buffer(topology=NVD2_topology):
            print(f"-----topology: {config.topology_name}, packet_size: {config.packet_size} chassis: {config.chassis}-----")
            print(f"Finish time: {time * 1000000} us")
            # 这是修改后的代码
            time_in_us = Decimal(str(time)) * Decimal('1000000')
            normalized_time = time_in_us.normalize()
            collective_time[connectivity][num_chunk][chunk] = f'{normalized_time} us'
            break
    trans_mat, pro_mat = build_rate_and_latency_matrices(NVD2_topology)
    print("trans_mat", trans_mat)
    print("pro_mat", pro_mat)
    # print(WAN_buffer)
    # print(NVD2_topology.nodes[11]['added_job'][(11, 10)])
    # print(NVD2_topology.nodes[2]['added_job'][(2, 10)])
    # print(list(NVD2_topology.predecessors(10)))

import numpy as np

def build_rate_and_latency_matrices(G):
    """
    G: 一个 networkx 图
       每条边上有 'link_capacity' (rate) 和 'propagation_latency' (时延) 属性
    返回:
       trans_mat: 二维 list，链路速率
       pro_mat  : 二维 list，传播时延
    """
    nodes = sorted(G.nodes())
    idx_map = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    # 初始化全 0 矩阵
    trans_mat = np.zeros((n, n), dtype=float)
    pro_mat = np.zeros((n, n), dtype=float)

    # 填充矩阵
    for u, v, data in G.edges(data=True):
        i, j = idx_map[u], idx_map[v]
        rate = data.get("link_capcapacity", 0)
        lat  = data.get("propagation_latency", 0)
        trans_mat[i][j] = rate
        trans_mat[j][i] = rate
        pro_mat[i][j]   = lat
        pro_mat[j][i]   = lat

    return trans_mat.tolist(), pro_mat.tolist()


if __name__ == "__main__":
    num_chunk_list = [1,2,4,8]
    chunk_size_list = [1,4,16,64,256]
    connectivity_list = [0.3,0.5,0.7,0.9]
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
                config.packet_size = Decimal(str(chunk))/ Decimal(str(1024.0))
                config.num_chunk = num_chunk
                config.connectivity = connectivity
                config.chassis = 2
                config.collective = 'ALLGATHER'
                config.topology_name = 'NVD2'
                config.connect_matrix = []
                policy= []
                main(collective_time, policy)

                # 记录结束时间
                end_time = time.time()

                # 计算并打印运行时间（单位：秒）
                execution_time = end_time - start_time
                execute_time[connectivity][num_chunk][chunk] += execution_time
                print(f"代码运行时间: {execution_time:.6f} 秒")
    print('collective_time:',collective_time)
    print('execute_time:',execute_time)
    print(policy)
