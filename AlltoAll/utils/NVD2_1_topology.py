import math
from decimal import Decimal

import matplotlib.pyplot as plt
import networkx as nx

import config


# 模拟 topologies.topology 模块中的 Topology 类
class Topology:
    def __init__(self):
        self.chunk_size = 1  # 假设 chunk_size 为 1
        self.chassis = config.chassis


class NDv2(Topology):
    def __init__(self):
        super().__init__()
        self.gpu_index = None
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
        if chassis == 2:
            capacity = [
                [0, 23, 46, 46, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [23, 0, 46, 23, 0, 46, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0],
                [46, 46, 0, 23, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [46, 23, 23, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0],
                [23, 0, 0, 0, 0, 23, 46, 46, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 46, 0, 0, 23, 0, 46, 23, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 23, 0, 46, 46, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 46, 46, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 46, 46, 23, 0, 0, 0],
                [107, 0, 0, 0, 0, 0, 0, 0, 23, 0, 46, 23, 0, 46, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 0, 23, 0, 0, 23, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 46, 23, 23, 0, 0, 0, 0, 46],
                [0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 23, 46, 46],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 23, 0, 46, 23],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 46, 46, 0, 23],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 23, 23, 0]
            ]
            self.gpu_index = {}
            for i in range(len(capacity)):
                self.gpu_index[i] = [i]
        else:
            self.switch_indices = [0]
            single_capacity = [
                [0, 23, 46, 46, 23, 0, 0, 0],
                [23, 0, 46, 23, 0, 46, 0, 0],
                [46, 46, 0, 23, 0, 0, 23, 0],
                [46, 23, 23, 0, 0, 0, 0, 46],
                [23, 0, 0, 0, 0, 23, 46, 46],
                [0, 46, 0, 0, 23, 0, 46, 23],
                [0, 0, 23, 0, 46, 46, 0, 23],
                [0, 0, 0, 46, 46, 23, 23, 0]
            ]
            capacity = [[0] * (8 * chassis + 1)]
            self.gpu_index = {}
            for i in range(len(capacity) - 1):
                self.gpu_index[i] = [i + 1]
            for i in range(chassis):
                for j in single_capacity:
                    cap = [0] * (8 * chassis)
                    for k in range(8):
                        cap[8 * i + k] = j[k]
                    capacity.append([0] + cap)
            for i in range(chassis):
                capacity[0][i * 8 + 1] = 107
                capacity[i * 8 + 2][0] = 107
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
                        row.append(1.3 * pow(10, -6))
                    else:
                        row.append(0.002)
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
        """for node in self.topology.nodes:
            print(node, self.topology.nodes[node]['memory'])"""

    def get_topology(self):
        G = nx.DiGraph()  # 使用有向图
        for node in self.nodes:
            if node not in self.switch_indices:
                G.add_node(node, memory=self.initial_buffer(node), type='GPU', job={}, added_job={}, receive={})
            else:
                G.add_node(node, memory=self.initial_buffer(node), type='switch', job={}, added_job={}, receive={})
        for i in range(len(self.capacity)):
            for j in range(len(self.capacity[i])):
                if self.capacity[i][j] == 25 or self.capacity[i][j] == 50:  # 如果有链路
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency=Decimal('0.0000007'),  # Decimal('0.0000007')
                               transmission_latency=self.packet_size / self.capacity[i][j],
                               state='free',
                               job=[],
                               type='NVlink',
                               total_time=self.packet_size / self.capacity[i][j],
                               connect=False)  # 添加有向边
                if self.capacity[i][j] == 12.5:
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency=Decimal('0.0000013'),  # Decimal('0.0000013')
                               transmission_latency=self.packet_size / self.capacity[i][j],
                               state='free',
                               job=[],
                               type='Switch',
                               total_time=self.packet_size / self.capacity[i][j],
                               connect=True)  # 添加有向边
        for node in self.nodes:
            for next_node in list(G.successors(node)):
                G.nodes[node]['job'][(node, next_node)] = []
                G.nodes[node]['added_job'][(node, next_node)] = []
                G.nodes[node][f'sender to {next_node}'] = []
            for pre_node in list(G.predecessors(node)):
                G.nodes[node][f'receiver from {pre_node}'] = []
        """G.nodes[1]['memory'][7] = {'buffer': 7, 'send_time ': 0, 'received_time ': 0}
        G.nodes[2]['memory'][7] = {'buffer': 7, 'send_time ': 0, 'received_time ': 0}"""
        if self.chassis != 2:
            mapping = {node: "switch" if node == 0 else node - 1 for node in G.nodes}

            # 重新命名节点，确保不丢失属性
            nx.relabel_nodes(G, mapping, copy=False)

        return G

    def initial_buffer(self, current_node):
        buffer = {}
        num_buffer = self.num_gpu * self.num_chunk + self.num_chunk * (self.num_gpu - 1)
        if self.chassis == 2:
            switch_num = 0
            self.switch_indices = []
        else:
            switch_num = 1
            self.switch_indices = [0]
        if self.chassis == 2:
            if current_node not in self.switch_indices:
                for buffer_index in range(num_buffer):
                    buffer[buffer_index] = {'buffer': None, 'dest_node': None}
                for buffer_index in range(self.num_gpu * self.num_chunk):
                    buffer[buffer_index] = {'buffer': current_node * self.num_gpu * self.num_chunk + buffer_index,
                                            'dest_node': math.floor(buffer_index/config.num_chunk)}
        else:
            if current_node not in self.switch_indices:
                for buffer_index in range(num_buffer):
                    buffer[buffer_index] = {'buffer': None, 'dest_node': None}
                for buffer_index in range(self.num_gpu * self.num_chunk):
                    buffer[buffer_index] = {
                        'buffer': (current_node - switch_num) * self.num_gpu * self.num_chunk + buffer_index,
                        'dest_node': math.floor(buffer_index/config.num_chunk)}

        """if config.collective == 'ALLGATHER':
            config.buffer_constant = 1
        else:
            config.buffer_constant = self.num_gpu
        for node in range(self.num_gpu * self.num_chunk * config.buffer_constant):
            buffer[node] = {'buffer': None, 'send_time': None, 'received_time': None}
        if self.chassis == 2:
            switch_num = 0
            self.switch_indices = []
        else:
            switch_num = 1
            self.switch_indices = [0]
        if current_node not in self.switch_indices:
            for i in range(self.num_chunk * config.buffer_constant):
                buffer[(current_node - switch_num) * self.num_chunk * config.buffer_constant + i] = {
                    'buffer': (current_node - switch_num) * self.num_chunk * config.buffer_constant + i, 'send_time': 0,
                    'received_time': 0}"""
        return buffer


if __name__ == '__main__':
    config.packet_size = Decimal(str(0.003906252))
    config.num_chunk = 4
    config.chassis = 1
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
