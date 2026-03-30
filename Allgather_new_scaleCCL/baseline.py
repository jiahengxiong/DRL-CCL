import bisect
import os
import sys
import time
from typing import Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
import networkx as nx


from decimal import Decimal
from Allgather_new_scaleCCL.utils.tools import add_node_job, select_node_job, start_send, combine_job, end_send, start_receive, end_receive, \
    check_buffer,queue, extract_gpu_gpu_policies
from Allgather_new_scaleCCL.utils.util import (
    load_topology,
    resolve_wan_dataset_dir,
    resolve_wan_model_dir,
    load_directional_streams,
    train_model_if_needed,
    update_wan_caps,
    load_wan_models_and_data,
)

from Allgather_new_scaleCCL.utils.simulate import simulate_policy_with_true_stream
import simpy
from Allgather_new_scaleCCL.path_ass import solve_time_indexed_milp

def build_topology():
    dc = load_topology(
        packet_size=config.packet_size,
        num_chunk=config.num_chunk,
        chassis=config.chassis,
        name=config.topology_name,
    )
    return dc, dc.topology

def update_wan_link_rates(topology, per_link_stream, datacenter, pos_map, advance):
    update_wan_caps(topology, per_link_stream, datacenter, pos_map=pos_map, advance=advance)

def build_path_ass_input(arrival_time, per_link_stream, topology, policy):
    flow_size = float(config.packet_size)
    DC1_BR = [0, 1]
    DC2_BR = [2, 3]
    BR_list = DC1_BR + DC2_BR
    for BR in BR_list:
        # 1) 定义 flows（单位：size 比特，arrival_time 秒，rate_limit bps）
        flows = []
        for chunk_id, src, dst, arrival in arrival_time:
            if dst == BR and src not in BR_list:
                flows.append({
                    "id": chunk_id,
                    "size": flow_size,
                    "arrival_time": float(arrival),
                    "rate_limit": float(topology[src][dst]["link_capcapacity"]),
                })
        # 2) 定义共享路径（capacity bps，delay 秒）
        shared_paths = []
        if BR in DC1_BR:
            dst_BR_list = DC2_BR
        else:
            dst_BR_list = DC1_BR
        for dst_BR in dst_BR_list:
            shared_paths.append({
                "id": dst_BR,
                "capacity": float(topology[BR][dst_BR]["link_capcapacity"]),
                "delay": float(topology[BR][dst_BR]["propagation_latency"]),
            })
        # 3) 为每个 flow（按索引 i）指定可用路径 id 集合
        flow_path_map = {}
        for i, f in enumerate(flows):
            flow_path_map[i] = dst_BR_list


        # 4) 时间参数（秒）
        sum_size = sum(f["size"] for f in flows)
        sum_cap = sum(p["capacity"] for p in shared_paths)
        sum_rmax = sum(f.get("rate_limit", 0.0) for f in flows)
        max_arrival = max(f["arrival_time"] for f in flows)
        effective_total_rate = min(sum_cap, sum_rmax) if (sum_cap > 0 and sum_rmax > 0) else max(sum_cap, sum_rmax, 1.0)
        lb_total = sum_size / effective_total_rate
        lb_flow = max(f["size"] / f.get("rate_limit", 1.0) for f in flows)
        time_horizon = max_arrival + 2.0 * max(lb_total, lb_flow)
        min_eff_rate = min(min(p["capacity"] for p in shared_paths), min(f.get("rate_limit", 1.0) for f in flows))
        time_slot_duration = max(1e-6, (min(f["size"] for f in flows) / min_eff_rate) / 50.0)

        # 5) 调用 MILP
        sol = solve_time_indexed_milp(
            flows=flows,
            shared_paths=shared_paths,
            flow_path_map=flow_path_map,
            time_horizon=time_horizon,
            time_slot_duration=time_slot_duration,
            use_fairness=True,
            time_limit=60,
            subslots_per_segment=5,
        )
        ass = sol["assignments"]

        for chunk_id in list(ass.keys()):
            nest_hop = ass[chunk_id]
            p = policy[chunk_id]
            for key, value in p.items():
                if len(value) > 2:
                    src = value[0]
                    switch_0 = value[1]
                    switch_1 = value[2]
                    dst = value[-1]
                    if switch_0 != BR:
                        print("ERROR! switch_0 != BR")
                    else:
                        policy[chunk_id][key] = [src, switch_0, nest_hop, dst]
    
    return policy
                        
                    



def decimal_range(start, stop, step):
    """
    使用 Decimal 模块实现高精度浮点数遍历。
    - start: 起始值（Decimal 类型）
    - stop: 终止值（Decimal 类型）
    - step: 步长（Decimal 类型）
    """
    current = Decimal(start)
    stop = Decimal(stop)
    step = Decimal(step)

    while current < stop:
        yield float(current)  # 将 Decimal 转换为 float 输出
        current += step


def build_broad_shallow_trees(graph, nodes):
    """
    Build a tree for each node as the root, ensuring breadth is maximized
    and depth is minimized.

    Parameters:
    - graph: NetworkX Graph representing the topology.
    - nodes: List of nodes in the graph.

    Returns:
    - dict: A dictionary where keys are root nodes and values are BFS trees (in NetworkX format).
    """
    BFS_trees = {}

    for root_node in nodes:
        # Perform BFS starting from the root node
        bfs_tree = nx.bfs_tree(graph, source=root_node)
        BFS_trees[root_node] = bfs_tree

    return BFS_trees


def broad_cast(tree, current_roots, visited, completion_times, packet_size):
    """
    Perform a single step of broadcast for the given tree with time simulation.

    Parameters:
    - tree: BFS tree (NetworkX DiGraph) rooted at the root node.
    - current_roots: List of nodes broadcasting in this step.
    - visited: Set of nodes that have already received the message.
    - completion_times: Dict tracking when each node finishes broadcasting.
    - packet_size: Size of the packet being sent (in GB).

    Returns:
    - next_roots: List of nodes that will broadcast in the next step.
    """
    next_roots = []
    for root in current_roots:
        root_time = completion_times[root]  # When the root finished its last broadcast
        for neighbor in tree.successors(root):
            if neighbor not in visited:
                # Get link properties
                edge_data = tree.get_edge_data(root, neighbor)
                bandwidth = edge_data["link_capcapacity"]  # GB/s
                propagation_latency = edge_data["propagation_latency"]  # seconds

                # Calculate transmission time and arrival time
                transmission_time = packet_size / bandwidth
                arrival_time = root_time + transmission_time + propagation_latency

                # Update completion time if earlier or unvisited
                if neighbor not in completion_times or arrival_time < completion_times[neighbor]:
                    completion_times[neighbor] = arrival_time

                visited.add(neighbor)
                next_roots.append(neighbor)
    return next_roots


def _build_test_stream_from_loader(path: str):
    data = load_snvang_dataset_by_day(path, train_ratio=0.8, scale_range=(0.01, 1.0))
    test_days = data["test_days"]  # 只用测试集
    xs = []
    for seg in test_days:
        x = seg.x
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        xs.append(x.astype(np.float32))
    if len(xs) == 0:
        raise ValueError("Empty test_days from load_snvang_dataset_by_day")
    return np.concatenate(xs, axis=0)  # 标准化后的 0.01~1.0 流


def _update_wan_links(topology: nx.DiGraph, packet_size: 'Decimal', cap_value: 'Decimal'):
    # 按“switch↔switch”识别 WAN 边；更新 link_capcapacity 与 transmission_latency
    if cap_value <= 0:
        cap_value = Decimal("1e-9")
    for u, v in topology.edges:
        if topology.nodes[u].get('type') == 'switch' and topology.nodes[v].get('type') == 'switch':
            e = topology.edges[(u, v)]
            e['link_capcapacity'] = cap_value
            e['transmission_latency'] = packet_size / cap_value
            if 'weight' in e:
                e['weight'] = e.get('propagation_latency', 0) + e['transmission_latency']


def global_policy(topology, datacenter, per_link_stream, per_link_pos):
    # topology = topology.copy()
    # nodes_to_process = [n for n in topology.nodes if topology.nodes[n].get('type') == 'switch']
    # for u in nodes_to_process:
    #     succs = list(topology.successors(u))
    #     switch_succs = [v for v in succs if topology.nodes[v].get('type') == 'switch']
    #     if len(switch_succs) > 1:
    #         best_v = None
    #         best_cap = None
    #         for v in switch_succs:
    #             try:
    #                 cap = float(topology.edges[(u, v)].get('link_capcapacity', 0))
    #             except Exception:
    #                 cap = 0.0
    #             if best_cap is None or cap > best_cap or (cap == best_cap and (best_v is None or v < best_v)):
    #                 best_cap = cap
    #                 best_v = v
    #         to_remove = [v for v in switch_succs if v != best_v]
    #         for v in to_remove:
    #             if topology.has_edge(u, v):
    #                 topology.remove_edge(u, v)
    node_list = topology.nodes()
    buffer_matrix = [[0 for _ in range(len(node_list) * config.num_chunk * config.buffer_constant)]
                     for _ in range(len(node_list) * config.num_chunk * config.buffer_constant)]
    sent_matrix = [[0 for _ in range(len(node_list) * config.num_chunk * config.buffer_constant)]
                   for _ in range(len(node_list) * config.num_chunk * config.buffer_constant)]
    DC_0_buffer, DC_1_buffer = [], []
    for node in node_list:
        memory = topology.nodes[node]['memory']
        for buffer_index, buffer in memory.items():
            if buffer['buffer'] is not None:
                buffer_matrix[node][buffer_index] = 1
                if topology.nodes[node]['DC'] == 0:
                    DC_0_buffer.append(buffer['buffer'])
                else:
                    DC_1_buffer.append(buffer['buffer'])
    # update_wan_link_rates(topology, per_link_stream, datacenter, pos_map=per_link_pos, advance=False)
    policy = []
    arrival_times = []
    time = Decimal(str(0.0))
    WAN_buffer = []
    while check_buffer(topology=topology) is False:
        # update_wan_link_rates(topology, per_link_stream, datacenter, pos_map=per_link_pos, advance=True)
        buffer_num_dict = {}
        for link in topology.edges:
            end_send(topology=topology, link=link, time=time, WAN_buffer=WAN_buffer)
        for link in topology.edges:
            start_receive(topology=topology, link=link, time=time)
        for link in topology.edges:
            end_receive(topology=topology, link=link, time=time)
        for node in node_list:
            add_node_job(
                topology=topology,
                src=node,
                time=time,
                memory_state=buffer_matrix,
                sent_matrix=sent_matrix,
                DC0=DC_0_buffer,
                DC1=DC_1_buffer,
                WAN_buffer=WAN_buffer,
                buffer_num_dict=buffer_num_dict
            )
        queue(topology=topology, memory_state=buffer_matrix, time=time)
        for node in node_list:
            start_send(
                topology=topology,
                node=node,
                time=time,
                memory_state=buffer_matrix,
                WAN_buffer=WAN_buffer,
                DC0=DC_0_buffer,
                DC1=DC_1_buffer,
                policy=policy,
                arrival_times=arrival_times
            )
        event_set = set()
        for link in topology.edges:
            job_list = topology.edges[link]['job']
            job_max = []
            removed_jobs = []
            time_list = []
            for job in job_list:
                for key in ['send_time', 'sent_time', 'receive_time', 'received_time']:
                    t = job[key]
                    time_list.append(t)
                    job_max.append(t)
                    if t > time:
                        event_set.add(t)
                if max(job_max) < time - (max(time_list) - min(time_list)):
                    removed_jobs.append(job)
            for job in removed_jobs:
                topology.edges[link]['job'].remove(job)
        event_list = list(event_set)
        sorted_event_list = sorted(event_list)
        index = bisect.bisect_right(sorted_event_list, time)
        if index < len(event_list):
            time = sorted_event_list[index]
    extracted = extract_gpu_gpu_policies(topology, policy)
    return extracted, arrival_times, time


def main(collective_time, chunk):
    """
    主函数逻辑分为四个清晰步骤：
    1. 加载数据集，并划分为训练集和测试集
    2. 用训练集训练神经网络
    3. 用测试集的第0个元素去求global_policy (初始化)
    4. 循环100个time step，用神经网络去单步预测下一步的WAN link值，并运行global_policy
    """
    # 步骤1 & 2: 加载数据集并训练模型 (load_wan_models_and_data 内部处理)
    dataset_dir, found = resolve_wan_dataset_dir()
    if found:
        print(f"WAN_link_dataset dir: {dataset_dir}")
    else:
        print(f"WAN_link_dataset not found, fallback dir: {dataset_dir}")
    model_dir = resolve_wan_model_dir()
    
    # wan_env: {(u, v): {'deployer': ..., 'test_data': ..., 'data_path': ...}}
    wan_env = load_wan_models_and_data(None, dataset_dir, model_dir)
    data_links = sum(1 for env in wan_env.values() if env.get('deployer') is not None)
    print(f"WAN links with dataset/model: {data_links}, fallback links: {len(wan_env) - data_links}")
    
    # 步骤3: 用测试集的第0个元素去求global_policy (初始化)
    print("Initializing global_policy with t=0 data...")
    current_stream = {}
    step_times_us = []
    
    # 准备初始流 (t=0)
    for pair, env in wan_env.items():
        test_data = env.get('test_data')
        if test_data is not None and len(test_data) > 0:
            val = float(test_data[0])
        else:
            val = 1.0
        # 使用 numpy 数组，确保 update_wan_caps 能读取到 size 属性
        current_stream[pair] = np.asarray([val], dtype=np.float32)
    
    # 构建拓扑并应用 t=0 的 WAN 速率
    datacenter, topology = build_topology()
    per_link_pos = {pair: 0 for pair in current_stream.keys()}
    update_wan_link_rates(topology, current_stream, datacenter, pos_map=per_link_pos, advance=False)
    policy, arrival_times, time_val = global_policy(topology, datacenter, current_stream, per_link_pos)
    print(f"Initial (t=0) Finish time: {time_val * 1000000} us")
    t0_us = Decimal(str(time_val)) * Decimal('1000000')
    # step_times_us.append(f"{t0_us.normalize()} us")
    
    # 步骤4: 循环100个time step，单步预测
    num_steps = 10
    print(f"Starting {num_steps} steps simulation loop...")
    comulative_time = 0
    
    for t in range(num_steps):
        next_stream = {}
        true_stream = {}
        
        for pair, env in wan_env.items():
            deployer = env.get('deployer')
            test_data = env.get('test_data')
            
            if deployer and test_data is not None and t < len(test_data):
                # 获取当前真实值 (t)
                x_t = float(test_data[t])
                x_next_t = float(test_data[t+1])
                # 预测下一步 (t+1)
                # continual_learning=True: 在预测后利用当前的 x_t 更新模型
                pred_next = deployer.predict(x_t, continual_learning=True)
                val = float(pred_next)
            else:
                val = 1.0
            
            # 使用预测值作为下一步的容量依据，确保传入 numpy 数组
            next_stream[pair] = np.asarray([val], dtype=np.float32)
            true_stream[pair] = np.asarray([x_next_t], dtype=np.float32)
        
        if t < 5:
            continue
        
        if (t + 1) % 10 == 0:
            vals = []
            for v in next_stream.values():
                try:
                    vals.append(float(v[0]))
                except Exception:
                    pass
            if vals:
                print(f"Step {t+1} stats -> min:{min(vals):.4f}, max:{max(vals):.4f}, mean:{(sum(vals)/len(vals)):.4f}")
            else:
                print(f"Step {t+1} stats -> no values")
            
        # 为每个 step 重新构建拓扑，应用预测流作为 WAN 速率
        datacenter, topology = build_topology()
        per_link_pos = {pair: 0 for pair in next_stream.keys()}
        update_wan_link_rates(topology, next_stream, datacenter, pos_map=per_link_pos, advance=False)
        # 运行 global_policy (基于预测值，内部按时间推进 advance=True)
        # policy, arrival_times, time_val = global_policy(topology, datacenter, next_stream, per_link_pos)
        # policy = build_path_ass_input(arrival_times, next_stream, topology, policy)
        t_sec = simulate_policy_with_true_stream(policy=policy, true_stream=true_stream, topology=topology)
        t_us = Decimal(str(t_sec)) * Decimal('1000000')
        comulative_time = comulative_time + float(t_us)
        step_times_us.append(f"{t_us.normalize()} us")

        
        if (t + 1) % 10 == 0:
            print(f"Step {t+1}/{num_steps}: Finish time: {time_val * 1000000} us")
    
    # 记录每一步的时间列表并在最后打印
    collective_time[config.connectivity][config.num_chunk][chunk] = step_times_us
    print("Collective times over steps (us):")
    print(step_times_us)
    print(f"Comulative time: {comulative_time} us")
    
    return policy, arrival_times

import numpy as np

def build_rate_and_latency_matrices(G):
    """
    G: 一个 networkx 图
       每条边上有 'link_capacity' (rate) 和 'propagation_latency' (时延) 属性
    返回:
       trans_mat: 二维 list，链路速率
       pro_mat  : 二维 list，传播时延
    """
    nodes = sorted(G.nodes())
    idx_map = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    # 初始化全 0 矩阵
    trans_mat = np.zeros((n, n), dtype=float)
    pro_mat = np.zeros((n, n), dtype=float)

    # 填充矩阵
    for u, v, data in G.edges(data=True):
        i, j = idx_map[u], idx_map[v]
        rate = data.get("link_capcapacity", 0)
        lat  = data.get("propagation_latency", 0)
        trans_mat[i][j] = rate
        trans_mat[j][i] = rate
        pro_mat[i][j]   = lat
        pro_mat[j][i]   = lat

    return trans_mat.tolist(), pro_mat.tolist()


if __name__ == "__main__":
    num_chunk_list = [1]
    chunk_size_list = [4]
    connectivity_list = [0.5]
    collective_time = {}
    execute_time = {}

    for connectivity in connectivity_list:
        collective_time[connectivity] = {}
        execute_time[connectivity] = {}
        for num_chunk in num_chunk_list:
            collective_time[connectivity][num_chunk] = {}
            execute_time[connectivity][num_chunk] = {}
            for chunk in chunk_size_list:
                collective_time[connectivity][num_chunk][chunk] = 0
                execute_time[connectivity][num_chunk][chunk] = 0
    for connectivity in connectivity_list:
        for num_chunk in num_chunk_list:
            for chunk in chunk_size_list:
                start_time = time.time()

                config.packet_size = Decimal(str(chunk))/ Decimal(str(1024.0))
                config.num_chunk = num_chunk
                config.connectivity = connectivity
                config.chassis = 2
                config.collective = 'ALLGATHER'
                config.topology_name = 'NVD2'
                config.connect_matrix = []
                policy, arrival_times = main(collective_time, chunk)

                end_time = time.time()
                execution_time = end_time - start_time
                execute_time[connectivity][num_chunk][chunk] += execution_time
                print(f"代码运行时间: {execution_time:.6f} 秒")
    # print('collective_time:',collective_time)
    # print('execute_time:',execute_time)
    # print("Policy (Start Send Times):", policy)
    # print("Arrival Times (Start Receive Times):", arrival_times)
