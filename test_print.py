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
nccl_topology = NCCL.nccl_topo(topology)

print("NCCL Ring Edges:")
for e in nccl_topology.edges():
    print(e)
