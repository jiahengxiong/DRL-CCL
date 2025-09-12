from decimal import Decimal

import config
from GNN_DRL.utils.util import load_topology
from utils.tools import simulate_allgather_pipeline_bfs, build_subchunk_weights_from_policy



def main(policy):
    datacenter = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk, chassis=config.chassis, name=config.topology_name)
    G = datacenter.topology
    GPU_list = []
    for node in G.nodes:
        if G.nodes[node]['type'] == 'GPU' or G.nodes[node]['type'] == 'gpu':
            GPU_list.append(node)

    # subchunk_weight_dict = build_subchunk_weights_from_policy(policy)
    # # print(subchunk_weight_dict)
    # reward = simulate_allgather_pipeline_bfs(G=G, packet_size_per_subchunk=config.packet_size / config.num_chunk,
    #                                          subchunk_priority_mats=subchunk_weight_dict, gpu_nodes=GPU_list,
    #                                          verbose=False)
    # print(reward)

    # Build simple list-based inputs from topology
    nodes = list(G.nodes())
    edges = []
    for u, v, data in G.edges(data=True):
        tx_lat = data.get('transmission_latency', None)
        prop_lat = data.get('propagation_latency', None)
        edges.append((u, v, tx_lat, prop_lat))

    # adjacency as dict of neighbor lists
    adjacency = {n: list(G.neighbors(n)) for n in G.nodes()}
    node_subchunks = {}
    for node in G.nodes:
        if node not in node_subchunks.keys():
            node_subchunks[node] = []
            for id, value in G.nodes[node]['memory'].items():
                if value['buffer'] is not None:
                    node_subchunks[node].append(value['buffer'])

    print("Nodes:", nodes)
    print("Edges (u, v, tx_lat, prop_lat):", edges[:10], "...")  # print first 10 for brevity
    print("Adjacency:", {k: adjacency[k] for k in list(adjacency)[:5]})  # preview
    print("Node subchunks:", node_subchunks)








if __name__ == '__main__':
    policy = [[0, 4, 0, Decimal('0.0')], [0, 4, 5, Decimal('0.0')], [0, 4, 6, Decimal('0.0')], [0, 4, 8, Decimal('0.0')], [1, 5, 1, Decimal('0.0')], [1, 5, 4, Decimal('0.0')], [1, 5, 6, Decimal('0.0')], [2, 6, 4, Decimal('0.0')], [2, 6, 5, Decimal('0.0')], [2, 6, 7, Decimal('0.0')], [3, 7, 0, Decimal('0.0')], [3, 7, 6, Decimal('0.0')], [3, 7, 8, Decimal('0.0')], [4, 8, 0, Decimal('0.0')], [4, 8, 4, Decimal('0.0')], [4, 8, 7, Decimal('0.0')], [4, 8, 9, Decimal('0.0')], [5, 9, 1, Decimal('0.0')], [5, 9, 8, Decimal('0.0')], [6, 10, 2, Decimal('0.0')], [6, 10, 11, Decimal('0.0')], [7, 11, 3, Decimal('0.0')], [7, 11, 10, Decimal('0.0')], [7, 11, 12, Decimal('0.0')], [7, 11, 15, Decimal('0.0')], [8, 12, 2, Decimal('0.0')], [8, 12, 11, Decimal('0.0')], [8, 12, 13, Decimal('0.0')], [9, 13, 2, Decimal('0.0')], [9, 13, 12, Decimal('0.0')], [9, 13, 14, Decimal('0.0')], [9, 13, 15, Decimal('0.0')], [10, 14, 13, Decimal('0.0')], [10, 14, 15, Decimal('0.0')], [11, 15, 3, Decimal('0.0')], [11, 15, 11, Decimal('0.0')], [11, 15, 13, Decimal('0.0')], [11, 15, 14, Decimal('0.0')], [4, 0, 2, Decimal('7E-7')], [3, 0, 3, Decimal('7E-7')], [5, 1, 2, Decimal('7E-7')], [1, 1, 3, Decimal('7E-7')], [9, 2, 0, Decimal('7E-7')], [8, 2, 1, Decimal('7E-7')], [11, 3, 0, Decimal('7E-7')], [7, 3, 1, Decimal('7E-7')], [2, 4, 1, Decimal('0.0000397625')], [4, 4, 5, Decimal('0.0000397625')], [4, 4, 6, Decimal('0.0000397625')], [1, 4, 8, Decimal('0.0000397625')], [3, 6, 4, Decimal('0.0000397625')], [3, 6, 5, Decimal('0.0000397625')], [1, 6, 7, Decimal('0.0000397625')], [2, 7, 8, Decimal('0.0000397625')], [5, 8, 4, Decimal('0.0000397625')], [5, 8, 7, Decimal('0.0000397625')], [0, 8, 9, Decimal('0.0000397625')], [8, 11, 10, Decimal('0.0000397625')], [6, 11, 12, Decimal('0.0000397625')], [6, 11, 15, Decimal('0.0000397625')], [9, 12, 11, Decimal('0.0000397625')], [7, 12, 13, Decimal('0.0000397625')], [10, 13, 12, Decimal('0.0000397625')], [8, 13, 14, Decimal('0.0000397625')], [8, 13, 15, Decimal('0.0000397625')], [10, 15, 11, Decimal('0.0000397625')], [7, 15, 14, Decimal('0.0000397625')], [10, 13, 2, Decimal('0.000078125')], [0, 0, 2, Decimal('0.000078825')], [2, 1, 2, Decimal('0.000078825')], [6, 2, 0, Decimal('0.000078825')], [0, 6, 7, Decimal('0.000078825')], [3, 8, 9, Decimal('0.000078825')], [11, 11, 10, Decimal('0.000078825')], [11, 11, 12, Decimal('0.000078825')], [5, 4, 5, Decimal('0.0000795250')], [5, 4, 6, Decimal('0.0000795250')], [6, 12, 13, Decimal('0.0000795250')], [6, 15, 14, Decimal('0.0000795250')], [1, 8, 9, Decimal('0.0001178875')], [9, 11, 10, Decimal('0.0001178875')], [10, 2, 0, Decimal('0.000156950')], [2, 8, 9, Decimal('0.000156950')], [10, 11, 10, Decimal('0.000156950')], [9, 0, 4, Decimal('0.0007507')], [11, 0, 7, Decimal('0.0007507')], [11, 0, 8, Decimal('0.0007507')], [8, 1, 4, Decimal('0.0007507')], [7, 1, 5, Decimal('0.0007507')], [7, 1, 9, Decimal('0.0007507')], [4, 2, 10, Decimal('0.0007507')], [5, 2, 12, Decimal('0.0007507')], [5, 2, 13, Decimal('0.0007507')], [5, 2, 15, Decimal('0.0007507')], [3, 3, 11, Decimal('0.0007507')], [3, 3, 15, Decimal('0.0007507')], [11, 0, 4, Decimal('0.000828825')], [6, 0, 7, Decimal('0.000828825')], [6, 0, 8, Decimal('0.000828825')], [7, 1, 4, Decimal('0.000828825')], [8, 1, 5, Decimal('0.000828825')], [8, 1, 9, Decimal('0.000828825')], [5, 2, 10, Decimal('0.000828825')], [0, 2, 12, Decimal('0.000828825')], [0, 2, 13, Decimal('0.000828825')], [0, 2, 15, Decimal('0.000828825')], [1, 3, 11, Decimal('0.000828825')], [1, 3, 15, Decimal('0.000828825')], [9, 4, 5, Decimal('0.000829525')], [8, 4, 6, Decimal('0.000829525')], [8, 4, 8, Decimal('0.000829525')], [7, 5, 6, Decimal('0.000829525')], [11, 7, 6, Decimal('0.000829525')], [11, 8, 9, Decimal('0.000829525')], [7, 9, 8, Decimal('0.000829525')], [4, 10, 11, Decimal('0.000829525')], [3, 11, 10, Decimal('0.000829525')], [3, 11, 12, Decimal('0.000829525')], [5, 12, 11, Decimal('0.000829525')], [5, 13, 14, Decimal('0.000829525')], [3, 15, 13, Decimal('0.000829525')], [3, 15, 14, Decimal('0.000829525')], [9, 4, 6, Decimal('0.0008685875')], [9, 4, 8, Decimal('0.0008685875')], [11, 6, 5, Decimal('0.0008692875')], [7, 6, 7, Decimal('0.0008692875')], [8, 8, 7, Decimal('0.0008692875')], [4, 11, 12, Decimal('0.0008692875')], [4, 11, 15, Decimal('0.0008692875')], [6, 0, 4, Decimal('0.000906950')], [9, 0, 7, Decimal('0.000906950')], [10, 0, 8, Decimal('0.000906950')], [0, 2, 10, Decimal('0.000906950')], [2, 2, 12, Decimal('0.000906950')], [4, 2, 13, Decimal('0.000906950')], [2, 2, 15, Decimal('0.000906950')], [6, 7, 6, Decimal('0.000907650')], [6, 8, 9, Decimal('0.000907650')], [1, 11, 10, Decimal('0.000907650')], [0, 12, 11, Decimal('0.000907650')], [0, 13, 14, Decimal('0.000907650')], [1, 15, 13, Decimal('0.000907650')], [1, 15, 14, Decimal('0.000907650')], [1, 11, 12, Decimal('0.0009083500')], [9, 8, 9, Decimal('0.0009467125')], [4, 15, 14, Decimal('0.0009467125')], [6, 6, 5, Decimal('0.0009474125')], [10, 0, 4, Decimal('0.000985075')], [10, 0, 7, Decimal('0.000985075')], [2, 2, 10, Decimal('0.000985075')], [2, 2, 13, Decimal('0.000985075')], [10, 8, 9, Decimal('0.000985775')], [2, 12, 11, Decimal('0.000985775')], [2, 15, 14, Decimal('0.000985775')], [10, 4, 5, Decimal('0.001063900')], [10, 4, 6, Decimal('0.001063900')]]
    config.chassis = 2
    config.num_chunk = 1
    config.packet_size = Decimal(1/1024)
    config.connectivity = 0.5
    config.collective = 'ALLGATHER'
    config.topology_name = 'NVD2'

    main(policy=policy)