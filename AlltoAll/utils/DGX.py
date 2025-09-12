from decimal import Decimal

import networkx as nx

import config




class Topology:
    def __init__(self):
        self.chunk_size = 1  # 假设 chunk_size 为 1
        self.switch_indices = []
        self.capacity = []
        self.topology = []
        self.alpha = []


class DGX(Topology):
    def __init__(self):
        super().__init__()
        self.chassis = config.chassis
        self.node_per_chassis = 16
        self.set_switch_indicies()
        self.construct_topology()

    def construct_topology(self):
        switch_connections = [0] + [1] * 16
        gpu_connections = [1] + [0] * 16
        single_node = [gpu_connections for _ in range(16)]
        single_node.insert(0, switch_connections)

        chassis = 2
        total_nodes = 16 * chassis + chassis
        self.topology = []
        for i in range(chassis):
            for j in single_node:
                cap = [0] * total_nodes
                for k in range(17):
                    cap[17 * i + k] = j[k]
                self.topology.append(cap)

        link_capacity = 125 / self.chunk_size
        self.capacity = [list(map(lambda x: x * link_capacity, r))
                         for r in self.topology]
        self.alpha = [list(map(lambda x: x * 0.35 * pow(10, -6), r))
                      for r in self.topology]

        inter_node_link_capacity = 12.5 / self.chunk_size
        taccl = {"1": [0], "3": [2], "5": [4], "7": [6], "9": [8], "11": [10], "13": [12], "15": [14]}
        for i in range(chassis):
            for j in range(chassis):
                if i == j:
                    continue
                for s in taccl:
                    for r in taccl[s]:
                        src = int(s) + i * 17 + 1
                        dst = r + j * 17 + 1
                        self.topology[src][dst] = 1
                        self.capacity[src][dst] = inter_node_link_capacity
                        self.alpha[src][dst] = 2.6 * pow(10, -6)

        # Print capacity matrix
        print("Capacity Matrix:")
        for row in self.capacity:
            print(row)

    def set_switch_indicies(self) -> None:
        self.switch_indices = [17 * i for i in range(2)]
        print(self.switch_indices)


class DGX2(DGX):
    def __init__(self, packet_size, num_chunk):
        super().__init__()
        print("Initializing DGX")
        self.num_chunk = num_chunk
        self.packet_size = Decimal(packet_size) / Decimal(str(self.num_chunk))

        self.capacity = [
            [Decimal(str(x)) for x in row] for row in self.capacity
        ]  # 转化为高精度
        self.nodes = list(range(len(self.capacity)))
        self.num_gpu = self.chassis * self.node_per_chassis

        self.topology = self.get_topology()
        for node in self.topology.nodes:
            print(node, self.topology.nodes[node]['memory'])

    def get_topology(self):
        G = nx.DiGraph()  # 使用有向图
        for node in self.nodes:
            if node not in self.switch_indices:
                G.add_node(node, memory=self.initial_buffer(node), type='GPU', job={}, added_job={})
            else:
                G.add_node(node, memory=self.initial_buffer(node), type='switch', job={}, added_job={})
        for i in range(len(self.capacity)):
            for j in range(len(self.capacity[i])):
                if self.capacity[i][j] == 125 or self.capacity[i][j] == 50:  # 如果有链路
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency=Decimal('0.00000035'),
                               transmission_latency=self.packet_size / self.capacity[i][j],
                               state='free',
                               job=[],
                               type='Switch',
                               total_time=self.packet_size / self.capacity[i][j],
                               connect=False)  # 添加有向边
                if self.capacity[i][j] == 12.5:
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency=Decimal('0.0000026'),
                               transmission_latency=self.packet_size / self.capacity[i][j],
                               state='free',
                               job=[],
                               type='Nvlink',
                               total_time=self.packet_size / self.capacity[i][j],
                               connect=True)  # 添加有向边
        for node in self.nodes:
            for next_node in list(G.successors(node)):
                G.nodes[node]['job'][(node, next_node)] = []
                G.nodes[node]['added_job'][(node, next_node)] = []
                G.nodes[node][f'sender to {next_node}'] = 'free'
            for pre_node in list(G.predecessors(node)):
                G.nodes[node][f'receiver from {pre_node}'] = 'free'
        """G.nodes[1]['memory'][7] = {'buffer': 7, 'send_time ': 0, 'received_time ': 0}
        G.nodes[2]['memory'][7] = {'buffer': 7, 'send_time ': 0, 'received_time ': 0}"""
        return G

    def initial_buffer(self, current_node):
        buffer = {}
        if config.collective == 'ALLGATHER':
            config.buffer_constant = 1
        else:
            config.buffer_constant = self.num_gpu
        for node in range(self.num_gpu * self.num_chunk * config.buffer_constant):
            buffer[node] = {'buffer': None, 'send_time': None, 'received_time': None}
        if current_node not in self.switch_indices:
            if current_node < 17:
                switch_num = 1
            else:
                switch_num = 2
            for i in range(self.num_chunk * config.buffer_constant):
                buffer[(current_node-switch_num) * self.num_chunk * config.buffer_constant + i] = {
                    'buffer': (current_node-switch_num) * self.num_chunk * config.buffer_constant + i, 'send_time': 0,
                    'received_time': 0}
        return buffer
