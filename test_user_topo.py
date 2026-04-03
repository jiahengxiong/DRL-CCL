import networkx as nx
import pickle
import Allgather_new_scaleCCL.utils.NCCL as NCCL
import importlib
importlib.reload(NCCL)

with open('/Users/xiongjiaheng/RDMA/CCL/TOPO/0.3/network_topology_0.pkl', 'rb') as f:
    G = pickle.load(f)

# Mock DC attribute since generate_topo might not add it
for n in G.nodes():
    if n in [4,5,6,7,8,9]:
        G.nodes[n]['DC'] = 0
        G.nodes[n]['type'] = 'GPU'
    elif n in [10,11,12,13,14,15]:
        G.nodes[n]['DC'] = 1
        G.nodes[n]['type'] = 'GPU'
    elif n in [0, 1]:
        G.nodes[n]['DC'] = 0
        G.nodes[n]['type'] = 'switch'
    elif n in [2, 3]:
        G.nodes[n]['DC'] = 1
        G.nodes[n]['type'] = 'switch'

nt = NCCL.nccl_topo(G)
print(sorted(list(nt.edges())))
