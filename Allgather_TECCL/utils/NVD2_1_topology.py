from decimal import Decimal

import matplotlib.pyplot as plt
import networkx as nx

import config
import os
import random
import glob
try:
    import simplejson as json
except ImportError:
    import json


# 模拟 topologies.topology 模块中的 Topology 类
class Topology:
    def __init__(self):
        self.chunk_size = 1  # 假设 chunk_size 为 1
        self.chassis = config.chassis


class NDv2(Topology):
    def __init__(self):
        super().__init__()
        self.node_per_chassis = 8
        self.construct_topology()

    def construct_topology(self):
        self.switch_indices = [0, 1, 2, 3]
        
        # Load topology from TOPO folder based on config.connectivity
        # Assuming file structure: RDMA/CCL/TOPO/{connectivity}/*.json
        
        # Determine paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up 2 levels to RDMA/CCL/
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        topo_folder = os.path.join(project_root, 'TOPO', str(config.connectivity))
        
        if not os.path.exists(topo_folder):
            raise FileNotFoundError(f"Topology directory not found: {topo_folder}")

        # Get list of json files
        json_files = glob.glob(os.path.join(topo_folder, '*.json'))
        if not json_files:
            raise FileNotFoundError(f"No JSON topology files found in {topo_folder}")
            
        # Select the first topology file deterministically
        json_files_sorted = sorted(json_files)
        selected_file = json_files_sorted[0]
        print(f"Loading topology from: {selected_file}")
        
        with open(selected_file, "r") as f:
            data = json.load(f, parse_float=Decimal, parse_int=Decimal)
            
        # Direct assignment as requested
        self.capacity = data["capacity"]
        self.pro = data["propagation"]
        # Use .get() with default to self.pro if float_propagation is missing
        self.TECCL_pro = data.get("float_propagation", self.pro)
        

class NVD2_1_topology(NDv2):
    def __init__(self, packet_size, num_chunk):
        super().__init__()
        # print("Initializing NVD2_1_topology")
        self.capacity = [
            [Decimal(str(x)) for x in row] for row in self.capacity
        ]  # 转化为高精度
        self.nodes = list(range(len(self.capacity)))
        self.packet_size = Decimal(str(packet_size)) / Decimal(str(num_chunk))
        self.num_chunk = num_chunk
        self.num_gpu = len(self.capacity) - 4

        self.topology = self.get_topology()
        # for node in self.topology.nodes:
        #     print(node, self.topology.nodes[node]['memory'])

    def get_topology(self):
        G = nx.DiGraph()  # 使用有向图
        DC_1 = [0, 1]
        DC_2 = [2, 3]
        # print(self.num_gpu / 2)
        DC_1_GPU = list(range(4, 4 + int(self.num_gpu / 2)))
        DC_2_GPU = list(range(4 + self.num_gpu, self.num_gpu + 4))
        self.DC_1 = DC_1_GPU + DC_1
        self.DC_2 = DC_2_GPU + DC_2

        for node in self.nodes:
            if node in DC_1_GPU + DC_1:
                DC = 0
            else:
                DC = 1
            if node not in self.switch_indices:
                G.add_node(node, memory=self.initial_buffer(node, DC), type='GPU', job={}, added_job={}, policy=[],
                           DC=DC, receive_buffer=[])
            else:
                G.add_node(node, memory=self.initial_buffer(node, DC), type='switch', job={}, added_job={}, policy=[],
                           in_queue=[], out_queue=[], DC=DC, receive_buffer=[], buffer_limitation=0, right=0, left=0,
                           served_gpu_once=set())
        
        for i in range(len(self.capacity)):
            for j in range(len(self.capacity[i])):
                if self.capacity[i][j] > 0:  # 如果有链路
                    # Determine link type based on connectivity to switches
                    is_switch_link = (i in self.switch_indices or j in self.switch_indices)
                    
                    if is_switch_link:
                        link_type = 'Switch'
                        connect = True
                    else:
                        link_type = 'NVlink'
                        connect = False

                    propagation_latency = self.pro[self.nodes[i]][self.nodes[j]]
                    transmission_latency = self.packet_size / self.capacity[i][j]
                    
                    G.add_edge(self.nodes[i], self.nodes[j],
                               link_capcapacity=self.capacity[i][j],
                               propagation_latency=propagation_latency,
                               transmission_latency=transmission_latency,
                               state='free',
                               job=[],
                               type=link_type,
                               weight=propagation_latency + transmission_latency,
                               num_chunk = 0,
                               connect=connect)

        # 新增功能：根据 self.chassis 删除节点 0 并调整其他节点编号
        if self.chassis == 1:
            if 0 in G:
                G.remove_node(0)
            mapping = {old_node: old_node - 1 for old_node in list(G.nodes)}
            G = nx.relabel_nodes(G, mapping, copy=True)
        # if self.chassis == 2:
        #     G.nodes[0]['type'] = 'switch'
        #     for node in list(G.nodes):
        #         G.nodes[node]['memory'].pop(0, None)
        for node in G.nodes:
            for next_node in list(G.successors(node)):
                G.nodes[node]['job'][(node, next_node)] = []
                G.nodes[node]['added_job'][(node, next_node)] = []
                G.nodes[node][f'sender to {next_node}'] = 'free'
                G.nodes[node][f'sender to {next_node} job'] = None
            for pre_node in list(G.predecessors(node)):
                G.nodes[node][f'receiver from {pre_node}'] = 'free'

        return G

    def initial_buffer(self, current_node, DC):
        buffer = {}
        if config.collective == 'ALLGATHER':
            config.buffer_constant = 1
        else:
            config.buffer_constant = self.num_gpu
        for node in range(self.num_gpu * self.num_chunk * config.buffer_constant):
            buffer[node] = {'buffer': None, 'send_time': None, 'received_time': None, 'DC': DC}
        if self.chassis == 2:
            switch_num = 0
            self.switch_indices = [0, 1, 2, 3]
        else:
            switch_num = 1
            self.switch_indices = [0, 1, 2, 3]

        switch_num = 4
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
    config.connectivity = 0.3
    topo = NVD2_1_topology(num_chunk=4, packet_size=1)
    G = topo.topology
    for node in G.nodes:
        print(node, G.nodes[node]['memory'])
    for node in G.nodes:
        if G.nodes[node]['type'] == 'switch':
            print(list(G.successors(node)), list(G.predecessors(node)))

    # 设置图形布局
    pos = nx.spring_layout(G)
