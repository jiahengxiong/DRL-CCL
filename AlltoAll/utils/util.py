import config
from AlltoAll.utils.NVD2_1_topology import NVD2_1_topology
from AlltoAll.utils.DGX import DGX2


def load_topology(packet_size, num_chunk, chassis, name):
    global topology
    if name == 'NVD2':
        topology = NVD2_1_topology(num_chunk=num_chunk, packet_size=packet_size)
    elif name == 'DGX':
        topology = DGX2(num_chunk=config.num_chunk, packet_size=config.packet_size)

    return topology
