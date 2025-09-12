import math
from decimal import Decimal

import matplotlib.pyplot as plt
import networkx as nx

import config
from test import capacity


# 模拟 topologies.topology 模块中的 Topology 类
class Topology:
    def __init__(self):
        self.chunk_size = 1  # 假设 chunk_size 为 1
        self.switch_indices = []
        self.chassis = config.chassis
        self.single_capacity = [
                [0, 23, 46, 46, 23, 0, 0, 0],
                [23, 0, 46, 23, 0, 46, 0, 0],
                [46, 46, 0, 23, 0, 0, 23, 0],
                [46, 23, 23, 0, 0, 0, 0, 46],
                [23, 0, 0, 0, 0, 23, 46, 46],
                [0, 46, 0, 0, 23, 0, 46, 23],
                [0, 0, 23, 0, 46, 46, 0, 23],
                [0, 0, 0, 46, 46, 23, 23, 0]
            ]


class NDv2(Topology):
    def __init__(self):
        super().__init__()
        self.node_per_chassis = 8
        self.construct_topology()

    def construct_topology(self):
        chassis = self.chassis

        # 定义 conversion_map，确保没有重复赋值
        conversion_map = {}
        conversion_map[0] = 0
        conversion_map[23] = 50 / self.chunk_size  # 使用最终赋值
        conversion_map[46] = 50 / (2 * self.chunk_size)
        conversion_map[107] = 12.5 / self.chunk_size


        self.switch_indices = []
        if chassis == 1:
            self.num_switch = 0
            self.num_gpu = 8
            self.num_node = 8
            capacity = self.single_capacity
        else:
            self.num_switch = 2 * chassis
            self.num_gpu = 8 * chassis
            self.num_node = self.num_switch + self.num_gpu
            switch_indices = list(range(self.num_switch))
            self.switch_indices = switch_indices
            capacity = [[0] * self.num_node for _ in range(self.num_node)]

            gpu_start = self.num_switch

            # inner chassis
            for i in range(chassis):
                start = i*8 + gpu_start
                end = start + 8
                for m in range(start, end):
                    for n in range(start, end):
                        capacity[m][n] = self.single_capacity[m - start][n - start]
            # print('123:',capacity)
            for i in range (self.num_switch):
                chassis_index = math.floor((i / 2))
                even = [0, 2, 4, 6]
                odd = [1,3,5,7]
                if i%2 == 0:
                    for e in even:
                        gpu_index = self.num_switch + chassis_index * 8 + e
                        print(i, chassis_index, gpu_index)
                        capacity[i][gpu_index] = 107
                        capacity[gpu_index][i] = 107
                else:
                    for o in odd:
                        gpu_index = self.num_switch + chassis_index * 8 + o
                        print(i, chassis_index, gpu_index)
                        capacity[i][gpu_index] = 107
                        capacity[gpu_index][i] = 107




            switch_connect = []
            # switch_connect = [(switch_indices[0], switch_indices[1])]
            for i in range(self.num_switch):
                if i%2 == 0:
                    switch_connect.append((switch_indices[i-1], switch_indices[i]))
            for connection in switch_connect:
                src = connection[0]
                dst = connection[1]
                capacity[src][dst] = 107
                capacity[dst][src] = 107



        capacity = list(map(list, zip(*capacity)))
        self.capacity = [list(map(lambda x: conversion_map[x], r))
                         for r in capacity]
        self.topology = [list(map(lambda x: int(x > 0), r))
                         for r in self.capacity]
        print("Capacity:", self.capacity)
        print("Topology:", self.topology)
        self.alpha = []
        for r in capacity:
            row = []
            for i in r:
                if i:
                    if i == 107:
                        row.append(0.0 * pow(10, -6))
                    else:
                        row.append(0.0 * pow(10, -6))
                else:
                    row.append(-1)
            self.alpha.append(row)
        print("Alpha:", self.alpha)

    def set_switch_indices(self) -> None:
        if self.chassis > 1:
            self.switch_indices = [0]


class NVD2_1_topology(NDv2):
    def __init__(self, packet_size, num_chunk):
        super().__init__()
        print("Initializing NVD2_1_topology")
        self.capacity = [
            [Decimal(str(x)) for x in row] for row in self.capacity
        ]  # 转化为高精度
        self.nodes = list(range(len(self.capacity)))
        self.packet_size = Decimal(str(packet_size)) / Decimal(str(num_chunk))
        self.num_chunk = num_chunk
        self.num_gpu = self.chassis * self.node_per_chassis

        self.topology = self.get_topology()
        for node in self.topology.nodes:
            print(node, self.topology.nodes[node]['memory'])

    def get_topology(self):
        G = nx.DiGraph()  # 使用有向图
        for node in self.nodes:
            chassis_index = math.floor((node / 10))
            if node not in self.switch_indices:
                G.add_node(node, memory=self.initial_buffer(node), type='GPU', job={}, added_job={}, policy = [], chassis = chassis_index)
            else:
                G.add_node(node, memory=self.initial_buffer(node), type='switch', job={}, added_job={}, policy = [], chassis = chassis_index)
        for i in range(len(self.capacity)):
            for j in range(len(self.capacity[i])):
                if self.capacity[i][j] == 25 or self.capacity[i][j] == 50:  # 如果有链路
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency= Decimal('0.0000007'),
                               transmission_latency=self.packet_size / self.capacity[i][j],
                               state='free',
                               job=[],
                               type='NVlink',
                               connect=False)
                if self.capacity[i][j] == 12.5:
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency=Decimal('0.0000013'),
                               transmission_latency=self.packet_size / self.capacity[i][j],
                               state='free',
                               job=[],
                               type='Switch',
                               connect=True)

                # 新增功能：根据 self.chassis 删除节点 0 并调整其他节点编号
        # if self.chassis == 1:
        #     if 0 in G:
        #         G.remove_node(0)
        #     mapping = {old_node: old_node - 1 for old_node in list(G.nodes)}
        #     G = nx.relabel_nodes(G, mapping, copy=True)
        # if self.chassis == 2:
        #     G.nodes[0]['type'] = 'switch'
        #     for node in list(G.nodes):
        #         G.nodes[node]['memory'].pop(0, None)
        for node in G.nodes:
            for next_node in list(G.successors(node)):
                G.nodes[node]['job'][(node, next_node)] = []
                G.nodes[node]['added_job'][(node, next_node)] = []
                G.nodes[node][f'sender to {next_node}'] = 'free'
            for pre_node in list(G.predecessors(node)):
                G.nodes[node][f'receiver from {pre_node}'] = 'free'



        return G

    def initial_buffer(self, current_node):
        buffer = {}
        if config.collective == 'ALLGATHER':
            config.buffer_constant = 1
        else:
            config.buffer_constant = self.num_gpu
        for node in range(self.num_gpu * self.num_chunk * config.buffer_constant):
            buffer[node] = {'buffer': None, 'send_time': None, 'received_time': None}
        if self.chassis == 1:
            switch_num = 0
        else:
            switch_num = 2 * self.chassis
        if current_node not in self.switch_indices:
            for i in range(self.num_chunk * config.buffer_constant):
                buffer[(current_node - switch_num) * self.num_chunk * config.buffer_constant + i] = {
                    'buffer': (current_node - switch_num) * self.num_chunk * config.buffer_constant + i, 'send_time': 0,
                    'received_time': 0}
        return buffer


if __name__ == '__main__':
    config.packet_size = Decimal(str(0.003906252))
    config.num_chunk = 1
    config.chassis = 2
    config.collective = 'ALLGATHER'
    config.topology_name = 'NVD2'
    config.connect_matrix = []
    topo = NVD2_1_topology(num_chunk=4, packet_size=1)
    G = topo.topology
    for node in G.nodes:
        print(node, G.nodes[node]['memory'])

    # 设置图形布局
    pos = nx.spring_layout(G)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

    # 绘制边
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrows=True)  # 添加箭头表示方向

    # 绘制节点标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # 绘制边的权重
    edge_labels = nx.get_edge_attributes(G, 'link_capcapacity')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # 显示图形
    plt.title("Directed Topology Visualization")
    plt.axis('off')
    plt.show()
