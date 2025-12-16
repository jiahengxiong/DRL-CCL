import math
import simplejson as json
from decimal import Decimal
import pickle

import networkx as nx
import random
import matplotlib.pyplot as plt

def gen_topo(conectivity, num_samples=1):

    # Your provided code to create the graph
    # NVlink_capacity = [46]  # No longer needed; capacities are now randomized per rules
    num_DC = 2
    switch_first_DC = [0, 1]
    switch_second_DC = [2, 3]
    GPUs_per_DC = 16
    mapping_GPU_indices = list(range(0, GPUs_per_DC * num_DC))
    G = nx.DiGraph()
    for i in switch_first_DC:
        for j in switch_second_DC:
            if i != j:
                # Inter-DC WAN links: capacity random between 10 and 100
                G.add_edge(i, j, type='WAN link', capacity=random.randint(10, 100), propagation_delay=Decimal('0.001'))
                G.add_edge(j, i, type='WAN link', capacity=random.randint(10, 100), propagation_delay=Decimal('0.001'))
    min_links = GPUs_per_DC + 1
    max_links = (GPUs_per_DC + 2) * (GPUs_per_DC + 1) * 0.5 - 1
    first_DC_GPUs = list(range(4, GPUs_per_DC + 4))
    second_DC_GPUs = list(range(GPUs_per_DC + 4, 2 * GPUs_per_DC + 4))
    first_DC_used_Link = []
    second_DC_used_Link = []
    first_DC_border_link = []
    second_DC_border_link = []

    for i in range(len(first_DC_GPUs) - 1):
        # Intra-DC NVlink: capacity random between 20 and 200
        G.add_edge(first_DC_GPUs[i], first_DC_GPUs[i + 1], type='NVlink', capacity=random.randint(20, 200),
                   propagation_delay=Decimal('0.0000007'))
        first_DC_used_Link.append((first_DC_GPUs[i], first_DC_GPUs[i + 1]))
        G.add_edge(first_DC_GPUs[i + 1], first_DC_GPUs[i], type='NVlink', capacity=random.randint(20, 200),
                   propagation_delay=Decimal('0.0000007'))
        first_DC_used_Link.append((first_DC_GPUs[i + 1], first_DC_GPUs[i]))

    for i in range(len(second_DC_GPUs) - 1):
        # Intra-DC NVlink: capacity random between 20 and 200
        G.add_edge(second_DC_GPUs[i], second_DC_GPUs[i + 1], type='NVlink', capacity=random.randint(20, 200),
                   propagation_delay=Decimal('0.0000007'))
        second_DC_used_Link.append((second_DC_GPUs[i], second_DC_GPUs[i + 1]))
        G.add_edge(second_DC_GPUs[i + 1], second_DC_GPUs[i], type='NVlink', capacity=random.randint(20, 200),
                   propagation_delay=Decimal('0.0000007'))
        second_DC_used_Link.append((second_DC_GPUs[i + 1], second_DC_GPUs[i]))

    # Border router links: capacity random between 20 and 200
    G.add_edge(switch_first_DC[0], first_DC_GPUs[0], type='Border router', capacity=random.randint(20, 200),
               propagation_delay=Decimal('0.0000007'))
    first_DC_border_link.append((switch_first_DC[0], first_DC_GPUs[0]))
    G.add_edge(first_DC_GPUs[0], switch_first_DC[0], type='Border router', capacity=random.randint(20, 200),
               propagation_delay=Decimal('0.0000007'))
    first_DC_border_link.append((first_DC_GPUs[0], switch_first_DC[0]))
    G.add_edge(switch_second_DC[0], second_DC_GPUs[0], type='Border router', capacity=random.randint(20, 200),
               propagation_delay=Decimal('0.0000007'))
    second_DC_border_link.append((switch_second_DC[0], second_DC_GPUs[0]))
    G.add_edge(second_DC_GPUs[0], switch_second_DC[0], type='Border router', capacity=random.randint(20, 200),
               propagation_delay=Decimal('0.0000007'))
    second_DC_border_link.append((second_DC_GPUs[0], switch_second_DC[0]))

    G.add_edge(switch_first_DC[-1], first_DC_GPUs[-1], type='Border router', capacity=random.randint(20, 200),
               propagation_delay=Decimal('0.0000007'))
    first_DC_border_link.append((switch_first_DC[-1], first_DC_GPUs[-1]))
    G.add_edge(first_DC_GPUs[-1], switch_first_DC[-1], type='Border router', capacity=random.randint(20, 200),
               propagation_delay=Decimal('0.0000007'))
    first_DC_border_link.append((first_DC_GPUs[-1], switch_first_DC[-1]))
    G.add_edge(switch_second_DC[-1], second_DC_GPUs[-1], type='Border router', capacity=random.randint(20, 200),
               propagation_delay=Decimal('0.0000007'))
    second_DC_border_link.append((switch_second_DC[-1], second_DC_GPUs[-1]))
    G.add_edge(second_DC_GPUs[-1], switch_second_DC[-1], type='Border router', capacity=random.randint(20, 200),
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
            if (i, j) not in second_DC_border_link and (j, i) not in second_DC_border_link:
                remain_links_second_DC_border[i].append((i, j))

    print(first_DC_used_Link)
    print(remain_links_first_DC)
    print(second_DC_used_Link)
    print(remain_links_second_DC)

    connectivity_list = [conectivity]
    import os
    # Create a directory for saving topologies based on connectivity value
    save_dir = f"/Users/xiongjiaheng/RDMA/CCL/TOPO/{conectivity}"
    os.makedirs(save_dir, exist_ok=True)
    for connectivity in connectivity_list:
        total_links_per_DC = math.floor(connectivity * max_links)
        add_links_num = total_links_per_DC - min_links
        all_first_candidates = remain_links_first_DC + remain_links_first_DC_border[0] + remain_links_first_DC_border[1]
        # 修正后的随机选择逻辑
        add_links_num_first = max(0, add_links_num)
        add_links_num_first = min(add_links_num_first, len(all_first_candidates))
        if add_links_num_first > 0:
            selected_first_links = random.sample(all_first_candidates, add_links_num_first)
        else:
            selected_first_links = []
        all_second_candidates = remain_links_second_DC + remain_links_second_DC_border[2] + remain_links_second_DC_border[3]
        # 修正后的随机选择逻辑
        add_links_num_second = max(0, add_links_num)
        add_links_num_second = min(add_links_num_second, len(all_second_candidates))
        if add_links_num_second > 0:
            selected_second_links = random.sample(all_second_candidates, add_links_num_second)
        else:
            selected_second_links = []
        for (u, v) in selected_first_links:
            if u in switch_first_DC or v in switch_first_DC:
                # Border link: capacity random between 20 and 200
                G.add_edge(u, v, type='border link', capacity=random.randint(20, 200), propagation_delay=Decimal('0.0000007'))
                G.add_edge(v, u, type='border link', capacity=random.randint(20, 200), propagation_delay=Decimal('0.0000007'))
            else:
                # Nv_link GPU-GPU: capacity random between 20 and 200
                G.add_edge(u, v, type='Nv_link', capacity=random.randint(20, 200),
                           propagation_delay=Decimal('0.0000007'))
                G.add_edge(v, u, type='Nv_link', capacity=random.randint(20, 200),
                           propagation_delay=Decimal('0.0000007'))
        for (u, v) in selected_second_links:
            if u in switch_second_DC or v in switch_second_DC:
                # Border link: capacity random between 20 and 200
                G.add_edge(u, v, type='border link', capacity=random.randint(20, 200), propagation_delay=Decimal('0.0000007'))
                G.add_edge(v, u, type='border link', capacity=random.randint(20, 200), propagation_delay=Decimal('0.0000007'))
            else:
                # Nv_link GPU-GPU: capacity random between 20 and 200
                G.add_edge(u, v, type='Nv_link', capacity=random.randint(20, 200),
                           propagation_delay=Decimal('0.0000007'))
                G.add_edge(v, u, type='Nv_link', capacity=random.randint(20, 200),
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

    for sample_idx in range(num_samples):
        # Re-randomize capacities on the fixed topology
        for u, v, data in G.edges(data=True):
            if data['type'] in ('NVlink', 'Nv_link', 'Border router', 'border link'):
                data['capacity'] = random.randint(20, 200)
            elif data['type'] == 'WAN link':
                data['capacity'] = random.randint(10, 100)

        import os
        pkl_path = os.path.join(save_dir, f"network_topology_{sample_idx}.pkl")
        with open(pkl_path, "wb") as f_pkl:
            pickle.dump(G, f_pkl)

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

        print(f"sample {sample_idx} capacity:\n{capacity}")
        print(f"sample {sample_idx} propagation:\n{propagation}")
        print(f"sample {sample_idx} float_propagation:\n{float_propagation}")
        print(max_links, total_links_per_DC, min_links)

        # 自定义 encoder，让 Decimal 转成字符串保存
        class DecimalEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Decimal):
                    return {"__decimal__": str(obj)}
                return super().default(obj)

        # 自定义 decoder，在读入时把 "__decimal__" 恢复成 Decimal 对象
        def decimal_hook(obj):
            if "__decimal__" in obj:
                return Decimal(obj["__decimal__"])
            return obj

        with open(os.path.join(save_dir, f"network_topology_{sample_idx}.json"), "w") as f:
            json.dump({
                "capacity": capacity,  # list
                "propagation": propagation,  # list[Decimal,...]
                "float_propagation": float_propagation  # list
            }, f, indent=4, use_decimal=True)

    # # 读取ni
    # with open("network_topology.json", "r") as f:
    #     loaded_data = json.load(f, object_hook=decimal_hook)
    #
    # print(loaded_data["propagation"][0], type(loaded_data["propagation"][0]))

if __name__ == '__main__':
    connectivity_list= [0.3, 0.5, 0.7, 0.9]
    for connectivity in connectivity_list:
        gen_topo(connectivity, num_samples=1000)