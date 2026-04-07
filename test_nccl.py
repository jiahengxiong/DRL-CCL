import networkx as nx
from Allgather_new_scaleCCL.utils.util import load_topology
import config
from decimal import Decimal

config.packet_size = Decimal('1')
config.num_chunk = 4
config.chassis = 1
config.topology_name = 'NVD2'

dc = load_topology(
    packet_size=config.packet_size,
    num_chunk=config.num_chunk,
    chassis=config.chassis,
    name=config.topology_name,
    propagation_latency=None
)

G = dc.topology

def get_cost(u, v, d):
    return float(d.get('propagation_latency', 0)) + float(d.get('transmission_latency', 0))

gpus = [n for n, d in G.nodes(data=True) if d.get('type') == 'GPU']
print("GPUs:", gpus)

