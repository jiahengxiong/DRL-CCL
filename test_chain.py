import networkx as nx
from Allgather_new_scaleCCL.utils.util import load_topology
import config
from decimal import Decimal
import Allgather_new_scaleCCL.utils.NCCL as NCCL

config.packet_size = Decimal('1')
config.num_chunk = 4
config.chassis = 1
config.topology_name = 'NVD2'
config.connectivity = 0.3
config.collective = 'ALLGATHER'

dc = load_topology(
    packet_size=config.packet_size,
    num_chunk=config.num_chunk,
    chassis=config.chassis,
    name=config.topology_name,
    propagation_latency=None
)

topology = dc.topology

def my_nccl_topo(topology):
    from collections import defaultdict
    dc_gpus = defaultdict(list)
    for node, data in topology.nodes(data=True):
        if data.get('type') == 'GPU':
            dc = data['DC']
            dc_gpus[dc].append(node)
    
    dc_keys = sorted(list(dc_gpus.keys()))
    
    def get_edge_cost(G, u, v):
        d = G.edges[u, v]
        return float(d.get('propagation_latency', 0)) + float(d.get('transmission_latency', 0))

    def get_all_cross_dc_paths(G, src, dst):
        allowed_nodes = {n for n, d in G.nodes(data=True) if d.get('type') == 'switch'}
        allowed_nodes.add(src)
        allowed_nodes.add(dst)
        
        subG = G.subgraph(allowed_nodes)
        paths = list(nx.all_simple_paths(subG, source=src, target=dst))
        path_costs = []
        for p in paths:
            cost = sum(get_edge_cost(subG, p[i], p[i+1]) for i in range(len(p)-1))
            path_costs.append((p, cost))
        
        path_costs.sort(key=lambda x: x[1])
        return path_costs

    paths1 = get_all_cross_dc_paths(topology, 8, 9)
    print("8 -> 9 paths:", paths1)
    paths2 = get_all_cross_dc_paths(topology, 14, 3)
    print("14 -> 3 paths:", paths2)

my_nccl_topo(topology)
