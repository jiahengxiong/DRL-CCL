import math
from decimal import Decimal

import networkx as nx
import random
import matplotlib.pyplot as plt

def gen_topo(conectivity):

    # Your provided code to create the graph
    NVlink_capacity = [46]
    num_DC = 2
    switch_first_DC = [0, 1]
    switch_second_DC = [2, 3]
    GPUs_per_DC = 6
    mapping_GPU_indices = list(range(0, GPUs_per_DC * num_DC))
    G = nx.DiGraph()
    for i in switch_first_DC:
        for j in switch_second_DC:
            if i != j:
                G.add_edge(i, j, type='WAN link', capacity=107, propagation_delay=Decimal('0.001'))
                G.add_edge(j, i, type='WAN link', capacity=107, propagation_delay=Decimal('0.001'))
    min_links = GPUs_per_DC + 1
    max_links = (GPUs_per_DC + 2) * (GPUs_per_DC + 1) * 0.5 - 1
    first_DC_GPUs = list(range(4, GPUs_per_DC + 4))
    second_DC_GPUs = list(range(GPUs_per_DC + 4, 2 * GPUs_per_DC + 4))
    first_DC_used_Link = []
    second_DC_used_Link = []
    first_DC_border_link = []
    second_DC_border_link = []

    for i in range(len(first_DC_GPUs) - 1):
        G.add_edge(first_DC_GPUs[i], first_DC_GPUs[i + 1], type='NVlink', capacity=random.choice(NVlink_capacity),
                   propagation_delay=Decimal('0.0000007'))
        first_DC_used_Link.append((first_DC_GPUs[i], first_DC_GPUs[i + 1]))
        G.add_edge(first_DC_GPUs[i + 1], first_DC_GPUs[i], type='NVlink', capacity=random.choice(NVlink_capacity),
                   propagation_delay=Decimal('0.0000007'))
        first_DC_used_Link.append((first_DC_GPUs[i + 1], first_DC_GPUs[i]))

    for i in range(len(second_DC_GPUs) - 1):
        G.add_edge(second_DC_GPUs[i], second_DC_GPUs[i + 1], type='NVlink', capacity=random.choice(NVlink_capacity),
                   propagation_delay=Decimal('0.0000007'))
        second_DC_used_Link.append((second_DC_GPUs[i], second_DC_GPUs[i + 1]))
        G.add_edge(second_DC_GPUs[i + 1], second_DC_GPUs[i], type='NVlink', capacity=random.choice(NVlink_capacity),
                   propagation_delay=Decimal('0.0000007'))
        second_DC_used_Link.append((second_DC_GPUs[i + 1], second_DC_GPUs[i]))

    G.add_edge(switch_first_DC[0], first_DC_GPUs[0], type='Border router', capacity=107,
               propagation_delay=Decimal('0.0000007'))
    first_DC_border_link.append((switch_first_DC[0], first_DC_GPUs[0]))
    G.add_edge(first_DC_GPUs[0], switch_first_DC[0], type='Border router', capacity=107,
               propagation_delay=Decimal('0.0000007'))
    first_DC_border_link.append((first_DC_GPUs[0], switch_first_DC[0]))
    G.add_edge(switch_second_DC[0], second_DC_GPUs[0], type='Border router', capacity=107,
               propagation_delay=Decimal('0.0000007'))
    second_DC_border_link.append((switch_second_DC[0], second_DC_GPUs[0]))
    G.add_edge(second_DC_GPUs[0], switch_second_DC[0], type='Border router', capacity=107,
               propagation_delay=Decimal('0.0000007'))
    second_DC_border_link.append((second_DC_GPUs[0], switch_second_DC[0]))

    G.add_edge(switch_first_DC[-1], first_DC_GPUs[-1], type='Border router', capacity=107,
               propagation_delay=Decimal('0.0000007'))
    first_DC_border_link.append((switch_first_DC[-1], first_DC_GPUs[-1]))
    G.add_edge(first_DC_GPUs[-1], switch_first_DC[-1], type='Border router', capacity=107,
               propagation_delay=Decimal('0.0000007'))
    first_DC_border_link.append((first_DC_GPUs[-1], switch_first_DC[-1]))
    G.add_edge(switch_second_DC[-1], second_DC_GPUs[-1], type='Border router', capacity=107,
               propagation_delay=Decimal('0.0000007'))
    second_DC_border_link.append((switch_second_DC[-1], second_DC_GPUs[-1]))
    G.add_edge(second_DC_GPUs[-1], switch_second_DC[-1], type='Border router', capacity=107,
               propagation_delay=Decimal('0.0000007'))
    second_DC_border_link.append((second_DC_GPUs[-1], switch_second_DC[-1]))

    remain_links_first_DC = []
    remain_links_second_DC = []
    for i in range(len(first_DC_GPUs)):
        for j in range(i + 1, len(first_DC_GPUs)):
            u = first_DC_GPUs[i]
            v = first_DC_GPUs[j]
            if (u, v) not in first_DC_used_Link and (v, u) not in first_DC_used_Link:
                remain_links_first_DC.append((u, v))
    for i in range(len(second_DC_GPUs)):
        for j in range(i + 1, len(second_DC_GPUs)):
            u = second_DC_GPUs[i]
            v = second_DC_GPUs[j]
            if (u, v) not in second_DC_used_Link and (v, u) not in second_DC_used_Link:
                remain_links_second_DC.append((u, v))

    remain_links_first_DC_border = {}
    remain_links_second_DC_border = {}
    for i in switch_first_DC:
        remain_links_first_DC_border[i] = []
        for j in first_DC_GPUs:
            if (i, j) not in first_DC_border_link and (j, i) not in first_DC_border_link:
                remain_links_first_DC_border[i].append((i, j))
    for i in switch_second_DC:
        remain_links_second_DC_border[i] = []
        for j in second_DC_GPUs:
            if (i, j) not in second_DC_border_link and (j, i) not in second_DC_GPUs:
                remain_links_second_DC_border[i].append((i, j))

    print(first_DC_used_Link)
    print(remain_links_first_DC)
    print(second_DC_used_Link)
    print(remain_links_second_DC)

    connectivity_list = [conectivity]
    for connectivity in connectivity_list:
        total_links_per_DC = math.floor(connectivity * max_links)
        add_links_num = total_links_per_DC - min_links
        selected_first_links = random.sample(remain_links_first_DC + remain_links_first_DC_border[0] + remain_links_first_DC_border[1] , add_links_num) # + random.sample(remain_links_first_DC_border[0], 1) + random.sample(remain_links_first_DC_border[1], 1)
        selected_second_links = random.sample(remain_links_second_DC + remain_links_second_DC_border[2] + remain_links_second_DC_border[3] , add_links_num) # + random.sample(remain_links_second_DC_border[2], 1) + random.sample(remain_links_second_DC_border[3], 1)
        for (u, v) in selected_first_links:
            if u in switch_first_DC or v in switch_first_DC:
                G.add_edge(u, v, type='border link', capacity=107, propagation_delay=Decimal('0.0000007'))
                G.add_edge(v, u, type='border link', capacity=107, propagation_delay=Decimal('0.0000007'))
            else:
                G.add_edge(u, v, type='Nv_link', capacity=random.choice(NVlink_capacity),
                           propagation_delay=Decimal('0.0000007'))
                G.add_edge(v, u, type='Nv_link', capacity=random.choice(NVlink_capacity),
                           propagation_delay=Decimal('0.0000007'))
        for (u, v) in selected_second_links:
            if u in switch_second_DC or v in switch_second_DC:
                G.add_edge(u, v, type='border link', capacity=107, propagation_delay=Decimal('0.0000007'))
                G.add_edge(v, u, type='border link', capacity=107, propagation_delay=Decimal('0.0000007'))
            else:
                G.add_edge(u, v, type='Nv_link', capacity=random.choice(NVlink_capacity),
                           propagation_delay=Decimal('0.0000007'))
                G.add_edge(v, u, type='Nv_link', capacity=random.choice(NVlink_capacity),
                           propagation_delay=Decimal('0.0000007'))

    # Visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold',
            arrowsize=20)
    # edge_labels = nx.get_edge_attributes(G, 'type')
    nx.draw_networkx(G, pos, font_color='red')
    plt.title("Network Visualization")
    plt.show()

    node_list = list(G.nodes())
    print(node_list)
    capacity = [[0] * len(node_list) for _ in range(len(node_list))]
    propagation = [[-1] * len(node_list) for _ in range(len(node_list))]

    for u in node_list:
        for v in node_list:
            if u != v and G.has_edge(u, v):
                capacity[u][v] = G.edges[u, v]['capacity']
                propagation[u][v] = G.edges[u, v]['propagation_delay']
    float_propagation = [[float(x) if isinstance(x, Decimal) else x for x in row] for row in propagation]

    print(capacity)
    print(propagation)
    print(float_propagation)
    print(max_links, total_links_per_DC, min_links)

if __name__ == '__main__':
    gen_topo(0.6)