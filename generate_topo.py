import math
import simplejson as json
from decimal import Decimal
import pickle

import networkx as nx
import random
import matplotlib.pyplot as plt


def _add_bidir_edge(graph, u, v, edge_type, capacity, propagation_delay):
    graph.add_edge(u, v, type=edge_type, capacity=capacity, propagation_delay=propagation_delay)
    graph.add_edge(v, u, type=edge_type, capacity=capacity, propagation_delay=propagation_delay)


def gen_topo(conectivity, num_samples=1):

    # Your provided code to create the graph
    # NVlink_capacity = [46]  # No longer needed; capacities are now randomized per rules
    num_DC = 2
    switch_first_DC = [0, 1]
    switch_second_DC = [2, 3]
    GPUs_per_DC = 6 # 9， 12， 15
    mapping_GPU_indices = list(range(0, GPUs_per_DC * num_DC))
    G = nx.DiGraph()
    for i in switch_first_DC:
        for j in switch_second_DC:
            if i != j:
                G.add_edge(i, j, type='WAN link', capacity=Decimal('12.5'), propagation_delay=Decimal('0.00005'))
                G.add_edge(j, i, type='WAN link', capacity=Decimal('12.5'), propagation_delay=Decimal('0.00005'))
    min_links = GPUs_per_DC + 4
    max_links = (GPUs_per_DC + 2) * (GPUs_per_DC + 1) * 0.5 - 1
    first_DC_GPUs = list(range(4, GPUs_per_DC + 4))
    second_DC_GPUs = list(range(GPUs_per_DC + 4, 2 * GPUs_per_DC + 4))
    first_DC_endpoint_GPUs = [first_DC_GPUs[0], first_DC_GPUs[-1]]
    second_DC_endpoint_GPUs = [second_DC_GPUs[0], second_DC_GPUs[-1]]
    first_DC_used_Link = []
    second_DC_used_Link = []

    for i in range(len(first_DC_GPUs)):
        u = first_DC_GPUs[i]
        v = first_DC_GPUs[(i + 1) % len(first_DC_GPUs)]
        _add_bidir_edge(G, u, v, 'NVlink', 50, Decimal('0.0000007'))
        first_DC_used_Link.append((u, v))
        first_DC_used_Link.append((v, u))

    for i in range(len(second_DC_GPUs)):
        u = second_DC_GPUs[i]
        v = second_DC_GPUs[(i + 1) % len(second_DC_GPUs)]
        _add_bidir_edge(G, u, v, 'NVlink', 50, Decimal('0.0000007'))
        second_DC_used_Link.append((u, v))
        second_DC_used_Link.append((v, u))

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
        total_links_per_DC = max(min_links, math.floor(connectivity * max_links))

        for gpu in first_DC_endpoint_GPUs:
            for sw in switch_first_DC:
                _add_bidir_edge(G, sw, gpu, 'border link', Decimal('12.5'), Decimal('0.0000007'))
        for gpu in second_DC_endpoint_GPUs:
            for sw in switch_second_DC:
                _add_bidir_edge(G, sw, gpu, 'border link', Decimal('12.5'), Decimal('0.0000007'))

        add_links_num_first = total_links_per_DC - min_links
        add_links_num_second = total_links_per_DC - min_links

        first_border_gpu_candidates = [gpu for gpu in first_DC_GPUs if gpu not in first_DC_endpoint_GPUs]
        second_border_gpu_candidates = [gpu for gpu in second_DC_GPUs if gpu not in second_DC_endpoint_GPUs]
        first_candidate_units = (
            [('gpu', edge, 1) for edge in remain_links_first_DC]
            + [('border', gpu, 2) for gpu in first_border_gpu_candidates]
        )
        second_candidate_units = (
            [('gpu', edge, 1) for edge in remain_links_second_DC]
            + [('border', gpu, 2) for gpu in second_border_gpu_candidates]
        )
        random.shuffle(first_candidate_units)
        random.shuffle(second_candidate_units)

        for kind, payload, cost in first_candidate_units:
            if cost > add_links_num_first:
                continue
            if kind == 'gpu':
                u, v = payload
                if G.has_edge(u, v):
                    continue
                _add_bidir_edge(G, u, v, 'Nv_link', 50, Decimal('0.0000007'))
            else:
                gpu = payload
                if any(G.has_edge(sw, gpu) for sw in switch_first_DC):
                    continue
                for sw in switch_first_DC:
                    _add_bidir_edge(G, sw, gpu, 'border link', Decimal('12.5'), Decimal('0.0000007'))
            add_links_num_first -= cost
            if add_links_num_first <= 0:
                break

        for kind, payload, cost in second_candidate_units:
            if cost > add_links_num_second:
                continue
            if kind == 'gpu':
                u, v = payload
                if G.has_edge(u, v):
                    continue
                _add_bidir_edge(G, u, v, 'Nv_link', 50, Decimal('0.0000007'))
            else:
                gpu = payload
                if any(G.has_edge(sw, gpu) for sw in switch_second_DC):
                    continue
                for sw in switch_second_DC:
                    _add_bidir_edge(G, sw, gpu, 'border link', Decimal('12.5'), Decimal('0.0000007'))
            add_links_num_second -= cost
            if add_links_num_second <= 0:
                break

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
        for u, v, data in G.edges(data=True):
            if data['type'] in ('NVlink', 'Nv_link', 'Border router', 'border link'):
                if data['type'] == 'border link':
                    data['capacity'] = Decimal('12.5')
                else:
                    data['capacity'] = 50
            elif data['type'] == 'WAN link':
                data['capacity'] = Decimal('12.5')

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
        gen_topo(connectivity, num_samples=1)
