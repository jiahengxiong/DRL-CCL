import config
from Allgather.utils.custom import select_node_job_refactored
# from config import WAN_buffer
# from generate_topo import switch_second_DC


def check_buffer(topology):
    flag = True
    for node in topology.nodes:
        if topology.nodes[node]['type'] == 'GPU':
            buffer_list = topology.nodes[node]['memory']
            for index, buffer in buffer_list.items():
                if buffer['buffer'] is None:
                    # print(f'Node {node} Buffer {index} is empty')
                    flag = False
                    break
    return flag


def end_receive(topology, link, time):
    src, dst = link
    link_job_list = topology.edges[link]['job']
    for link_job in link_job_list:
        if time >= link_job['received_time']:
            topology.nodes[dst][f'receiver from {src}'] = 'free'
            sent_buffer = link_job['buffer']
            topology.nodes[dst]['memory'][sent_buffer]['buffer'] = sent_buffer
            topology.nodes[dst]['memory'][sent_buffer]['received_time'] = time

            # ✅ 正确按 buffer id 清理 connect_matrix
            if topology.edges[link]['connect']:
                buf = link_job.get('buffer')
                if buf is not None and buf in config.connect_matrix:
                    config.connect_matrix.remove(buf)


def start_receive(topology, link, time):
    src = link[0]
    dst = link[1]
    link_job_list = topology.edges[link]['job']
    flag = False
    for link_job in link_job_list:
        if time >= link_job['receive_time']:
            flag = True
            topology.nodes[dst][f'receiver from {src}'] = 'busy'
            if topology.nodes[dst]['type'] == 'switch':
                if link_job not in topology.nodes[dst]['receive_buffer']:

                    topology.nodes[dst]['receive_buffer'].append(link_job)
                    # break
            # break
            # if topology.nodes[dst]['type'] == 'switch':
            #     topology.nodes[dst]['buffer_limitation'] += 1
            # break
    # if flag:
    #     # if topology.nodes[dst]['type'] == 'switch':
    #         if topology.nodes[dst]['type'] == 'switch' and topology.nodes[src]['type'] == 'GPU':
    #             topology.nodes[dst]['right'] += 1
    #         if topology.nodes[dst]['type'] == 'switch' and topology.nodes[src]['type'] == 'switch':
    #             topology.nodes[dst]['left'] += 1


def end_send(topology, link, time, WAN_buffer):
    src, dst = link
    link_job_list = topology.edges[link]['job']
    max_sent_time, job = 0, None
    for link_job in link_job_list:
        if link_job['sent_time'] > max_sent_time:
            max_sent_time = link_job['sent_time']
            job = link_job['buffer']

    if time >= max_sent_time:
        topology.nodes[src][f'sender to {dst}'] = 'free'
        topology.nodes[src][f'sender to {dst} job'] = None

        # ✅ 这里在“发完”时就释放 connect_matrix（或你也可只在 end_receive 释放）
        if topology.edges[link]['connect'] and job is not None and job in config.connect_matrix:
            config.connect_matrix.remove(job)

import random
def start_send(topology, node, time, memory_state, WAN_buffer, DC0, DC1,policy):
    jobs_list = topology.nodes[node]['job']
    successors = list(topology.successors(node))
    # num_nodes = len(topology.nodes)
    # DC_0_GPUs = list(range(4, 4 + int((num_nodes - 4) / 2)))
    # DC_1_GPUs = list(range(4 + int((num_nodes - 4) / 2), num_nodes))
    for dst in successors:

        if topology.nodes[node][f'sender to {dst}'] == 'free':
            selected_job = select_node_job(topology=topology, dst=dst, time=time, node=node)
            # selected_job = select_node_job_refactored(topology, dst, time, node, config.connect_matrix)
            # jobs = jobs_list[(node, dst)]
            if len(selected_job) > 0:
                job = selected_job[0]
                # job = jobs[0]
                link = topology.edges[node, dst]
                transmission_latency = link['transmission_latency']
                propagation_latency = link['propagation_latency']
                topology.edges[node, dst]['job'].append(
                    {'buffer': job['buffer'], 'send_time': time, 'sent_time': time + transmission_latency,
                     'receive_time': time + propagation_latency,
                     'received_time': time + transmission_latency + propagation_latency})
                # print(job, topology.nodes[node]['job'][(node, dst)])
                topology.nodes[node]['job'][(node, dst)].remove(job)
                topology.nodes[node][f'sender to {dst}'] = 'busy'
                topology.nodes[node][f'sender to {dst} job'] = {'buffer': job['buffer'], 'send_time': time, 'sent_time': time + transmission_latency,
                     'receive_time': time + propagation_latency,
                     'received_time': time + transmission_latency + propagation_latency}
                link_type = topology.edges[node, dst]['type']
                if topology.nodes[dst]['type'] == 'switch':
                    if job['buffer'] not in WAN_buffer:
                        WAN_buffer.append(job['buffer'])
                # elif dst in DC_0_GPUs:
                #     DC0.append(job['buffer'])
                # elif dst in DC_1_GPUs:
                #     DC1.append(job['buffer'])
                policy.append([job['buffer'], node, dst, time])
                print(
                    f"In time {time}, node {node} sent buffer {job} to node {dst} will start receive at {time + transmission_latency}, finish received at {time + transmission_latency + propagation_latency} via {link_type}")
                memory_state[dst][job['buffer']] = 1
                if dst == 1:
                    memory_state[0][job['buffer']] = 1
                if dst == 0:
                    memory_state[1][job['buffer']] = 1
                if dst == 2:
                    memory_state[3][job['buffer']] = 1
                if dst == 3:
                    memory_state[2][job['buffer']] = 1
                # num_nodes = len(topology.nodes)
                # DC_0_GPUs = list(range(4, 4 + int((num_nodes - 4) / 2)))
                # DC_1_GPUs = list(range(4 + int((num_nodes - 4) / 2), num_nodes))
                # if topology.nodes[node]['type'] == 'switch' and dst in DC_0_GPUs:
                #     for i in DC_0_GPUs:
                #         pro  = list(topology.predecessors(i))
                #         if 0 in pro or 1 in pro:
                #             memory_state[i][job['buffer']] = 1
                # if topology.nodes[node]['type'] == 'switch' and dst in DC_1_GPUs:
                #     for i in DC_1_GPUs:
                #         pro  = list(topology.predecessors(i))
                #         if 2 in pro or 3 in pro:
                #             memory_state[i][job['buffer']] = 1

                # if topology.nodes[dst]['type'] == 'switch':
                #     if topology.nodes[dst]['type'] == 'switch' and topology.nodes[node]['type'] == 'GPU':
                #         topology.nodes[dst]['right'] += 1
                #     if topology.nodes[dst]['type'] == 'switch' and topology.nodes[node]['type'] == 'switch':
                #         topology.nodes[dst]['left'] += 1
                # WAN_buffer.append(job['buffer'])

                # if topology.nodes[node]['memory'][job['buffer']]['buffer'] is None:
                    # print(
                    #     f"ERROR! ! ! The node {node} want to sent buffer {job} to node {dst}! But is None in node {node}! ")

    # print(jobs)
    """for (src, dst), job in jobs.items():
        if len(job) > 0:
            num_packet = Decimal(len(job))
            link = topology.edges[src, dst]
            transmission_latency = link['transmission_latency'] * num_packet
            propagation_latency = link['propagation_latency']
            topology.edges[src, dst]['job'].append(
                {'buffer': job, 'send_time': time, 'sent_time': time + transmission_latency,
                 'receive_time': time + propagation_latency,
                 'received_time': time + transmission_latency + propagation_latency})
            topology.nodes[node]['job'][(src, dst)] = []
            topology.nodes[node][f'sender to {dst}'] = 'busy'
            print(f"In time {time}, node {src} sent buffer {job} to node {dst}")
            for sent_buffer in job:
                if memory_state[dst][sent_buffer] == 0:
                    memory_state[dst][sent_buffer] = 1
                    topology.nodes[node]['memory'][sent_buffer]['send_time'] = time
                else:
                    print(f"ERROR!!! {src} want to send buffer {sent_buffer} to {dst}")"""
    return


def combine_job(topology, node):
    job = topology.nodes[node]['job']
    for (src, dst), memory in job.items():
        combined_buffer = []
        for buffer in memory:
            combined_buffer.append(buffer['buffer'])
        topology.nodes[node]['job'][(src, dst)] = combined_buffer

def dedup_keep_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def select_node_job(topology, dst, time, node):
    dst_list = []
    num_nodes = len(topology.nodes)
    DC_0_GPUs = list(range(4, 4 + int((num_nodes - 4) / 2)))
    DC_1_GPUs = list(range(4 + int((num_nodes - 4) / 2), num_nodes))
    if dst == 1 or dst == 0:
        predecessors = list(topology.predecessors(0)) + list(topology.predecessors(1))
        dst_list = [0,1]

    elif dst == 2 or dst == 3:
        predecessors = list(topology.predecessors(2)) + list(topology.predecessors(3))
        dst_list = [2,3]
    else:
        # print(DC_0_GPUs, DC_1_GPUs)
        if dst in DC_0_GPUs:
            if node in [0, 1]:
                predecessors = list(topology.predecessors(dst)) + [0, 1]
                for i in DC_0_GPUs:
                    if 0 in topology.predecessors(i) or 1 in topology.predecessors(i):
                        dst_list.append(i)
            else:
                predecessors = list(topology.predecessors(dst))
                dst_list = [dst]
        elif dst in DC_1_GPUs:
            if node in [2, 3]:
                predecessors = list(topology.predecessors(dst)) + [2, 3]
                for i in DC_1_GPUs:
                    if 2 in topology.predecessors(i) or 3 in topology.predecessors(i):
                        dst_list.append(i)
            else:
                predecessors = list(topology.predecessors(dst))
                dst_list = [dst]
    predecessors = dedup_keep_order(predecessors)
    dst_list = dedup_keep_order(dst_list)

    # predecessors = list(topology.predecessors(dst))

    jobs = {}
    estimated_time = {}
    select_buffer = {}

    for src in predecessors:
        # jobs[src] = []
        for dst_i in dst_list:
            jobs[(src, dst_i)] = []
            if (src, dst_i) not in topology.nodes[src]['job'].keys():
                continue
            job = topology.nodes[src]['job'][(src, dst_i)]
            # print('job',src,dst_i, job)
            estimated_time[(src, dst_i)] = 0
            link_jobs = topology.edges[src, dst_i]['job']
            for link_job in link_jobs:
                if link_job["sent_time"] - time > estimated_time[(src, dst_i)]:
                    estimated_time[(src, dst_i)] = link_job["sent_time"] - time
            for buffer in job:
                jobs[(src, dst_i)].append(buffer['buffer'])

    for (src, dst_i), buffers in jobs.items():
        for buffer in buffers:
            only = True
            for (checked_src, checked_dst_i), checked_buffers in jobs.items():
                if (src, dst_i) != (checked_src, checked_dst_i):
                    if buffer in checked_buffers:
                        only = False
                        if buffer not in select_buffer.keys():
                            select_buffer[buffer] = []
            if only is True:
                estimated_time[(src, dst_i)] += topology.edges[src, dst_i]['transmission_latency']
            else:
                select_buffer[buffer].append((src, dst_i))

    # if dst == 1 or dst == 0:
    #     print(dst, select_buffer)

    # print(jobs, select_buffer, estimated_time)
    for buffer, src_dst_list in select_buffer.items():
        for (src, dst_i) in src_dst_list:
            if buffer in jobs[(src, dst_i)]:
                """print(
                    f"For node {dst}, In node {src}, buffer {buffer} should be remove because it exists in {src}")"""
                jobs[(src, dst_i)].remove(buffer)
    for buffer, src_dst_list in select_buffer.items():
        src_time = {}
        for (src, dst_i) in src_dst_list:
            src_time[(src, dst_i)] = estimated_time[(src, dst_i)] + topology.edges[src, dst_i]['transmission_latency']

        min_src_dst_i = min(src_dst_list, key=lambda src_find: src_time[src_find])

        jobs[min_src_dst_i].append(buffer)
        estimated_time[min_src_dst_i] += topology.edges[min_src_dst_i]['transmission_latency']
    buffers = jobs[(node, dst)]
    # print(jobs)
    job = None
    if len(buffers) == 0:
        return []
    else:
        if topology.edges[node, dst]['connect']:
            for job in buffers:
                if job not in config.connect_matrix:
                    config.connect_matrix.append(job)
                    break
        else:
            job = buffers[0]
        if job is None:
            return []
        for src in predecessors:
            if src != node:
                # print(topology.nodes[src]['job'][(src, dst)], {'buffer': job})
                for dst_i in dst_list:
                    if (src, dst_i) not in topology.nodes[src]['job'].keys():
                        continue
                    if {'buffer': job} in topology.nodes[src]['job'][(src, dst_i)]:
                        # print("remove job")
                        topology.nodes[src]['job'][(src, dst_i)].remove({'buffer': job})
        # R = []
        # for j in buffers:
        #     R.append({'buffer': j})
        return [{'buffer':job}]
from collections import defaultdict
import heapq

def deduplicate_balanced(input_dict):
    """
    对输入字典中的元素进行去重，确保每个元素在最终结果的整体中只出现一次。
    同时，在处理重复元素时，会优先分配给当前列表长度最短的键，以实现均衡。
    元素不会被移动到它原始不存在的键的列表中。

    Args:
        input_dict (dict): 键为字符串，值为包含元素的列表的字典。
                           示例: {'a': [1, 2, 3], 'b': [2, 3, 4], 'c': [1, 5]}

    Returns:
        dict: 一个新的字典，其中元素已去重，并尽可能均衡地分配到其原始归属的键中。
              示例: {'a': [1, 5], 'b': [2, 4], 'c': [3]} (元素的顺序可能不同，均衡结果可能略有差异)
    """

    # # 1. 记录每个元素出现在哪些键中
    # # {元素: [key1, key2, ...]}
    #
    # # 打乱顺序
    # items = list(input_dict.items())
    # random.shuffle(items)
    #
    # # 构建新字典
    # input_dict = dict(items)
    element_to_keys = defaultdict(list)
    for key, lst in input_dict.items():
        # 对每个键的原始列表先进行内部去重，避免同一个键内部的重复干扰后续判断
        # 这一步是确保最终结果中，每个元素即使在某个键下出现多次，也只处理一次其"归属"问题
        for item in set(lst): # 使用set进行内部去重
            element_to_keys[item].append(key)

    # 2. 初始化结果字典和跟踪每个键的当前长度
    result = {k: [] for k in input_dict}
    key_lengths = {k: 0 for k in input_dict}

    # 3. 为每个键创建一个优先级队列（小顶堆）来跟踪其当前长度
    # 堆中存储 (当前长度, 键的索引, 键名)
    # 键的索引用于在长度相同时进行tie-break，确保稳定性或避免总选字典序小的
    key_index = {k: idx for idx, k in enumerate(input_dict.keys())}# 确保索引一致性

    # 4. 遍历所有元素，进行分配
    # 排序元素以确保每次运行结果一致（如果顺序不重要可以不排序）
    for item in sorted(element_to_keys.keys()):
        possible_keys = element_to_keys[item]

        if len(possible_keys) == 1:
            # 如果元素只存在于一个键中，直接分配给它
            chosen_key = possible_keys[0]
            result[chosen_key].append(item)
            key_lengths[chosen_key] += 1
        else:
            # 如果元素存在于多个键中，找到这些键中当前列表最短的那个
            # 使用一个小顶堆来选择最佳键 (长度, 键的索引, 键名)
            candidates_heap = []
            for k in possible_keys:
                heapq.heappush(candidates_heap, (key_lengths[k], key_index[k], k))

            # 弹出最短的键
            _, _, chosen_key = heapq.heappop(candidates_heap)

            # 将元素添加到 chosen_key 的结果列表中
            result[chosen_key].append(item)
            key_lengths[chosen_key] += 1

    return result

def add_node_job(topology, src, time, memory_state, sent_matrix, DC0, DC1, WAN_buffer,buffer_num_dict):
    # print(time)
    DC_0_switch = [0,1]
    DC_1_switch = [2,3]
    num_GPU = len(list(topology.nodes)) - 4
    DC_0_GPU = list(range(4,4+int(num_GPU/2)))
    DC_1_GPU = list(range(4 + int(num_GPU / 2), num_GPU + 4))
    memory = topology.nodes[src]['memory']
    successors = list(topology.successors(src))
    successors_switch = []
    successors_GPU = []
    Type = topology.nodes[src]['type']
    for successor in successors:
        if topology.nodes[successor]['type'] == 'GPU':
            successors_GPU.append(successor)
        else:
            successors_switch.append(successor)
    """print("memory:", memory)
    print("successors:", successors)
    print("job:", topology.nodes[src]['job'])"""
    if Type == 'GPU':
        switch_job = {}
        for buffer_index, buffer in memory.items():
            # link = topology.edges[src, dst]
            # print("link:", link)
            if buffer['buffer'] is not None:
                for dst in successors:
                    if topology.nodes[dst]['type'] == 'GPU':
                        if topology.nodes[dst]['type'] == 'GPU':
                            if memory_state[dst][buffer_index] == 0:
                                if {'buffer': buffer['buffer']} not in topology.nodes[src]['job'][(src, dst)]:
                                    topology.nodes[src]['job'][(src, dst)].append({'buffer': buffer['buffer']})
                                    topology.nodes[src]['added_job'][(src, dst)].append(buffer_index)
                        # else:
                        #     # if buffer['buffer'] not in WAN_buffer:
                        #         if memory_state[dst][buffer_index] == 0 and buffer_index not in \
                        #                 topology.nodes[src]['added_job'][(src, dst)]:
                        #             if buffer_index not in topology.nodes[src]['added_job'][(src, dst)]:
                        #                 topology.nodes[src]['job'][(src, dst)].append({'buffer': buffer['buffer']})
                        #                 topology.nodes[src]['added_job'][(src, dst)].append(buffer_index)
                    else:
                        if src in DC_0_GPU and buffer['buffer'] not in DC0:
                            continue
                        elif src in DC_1_GPU and buffer['buffer'] not in DC1:
                            continue
                        if buffer['buffer'] in WAN_buffer:
                            continue
                        # else:
                        #     print('buffer[buffer]', src, buffer['buffer'], WAN_buffer)

                        if memory_state[dst][buffer_index] == 0:
                            # if buffer_index not in topology.nodes[src]['added_job'][(src, dst)]:
                                if (src, dst) not in switch_job.keys():
                                    switch_job[(src, dst)] = []
                                    switch_job[(src, dst)].append(buffer['buffer'])
                                else:
                                    if buffer['buffer'] not in switch_job[(src, dst)]:
                                        switch_job[(src, dst)].append(buffer['buffer'])
        busy_state = 0
        for (src, dst), value in switch_job.items():
            if topology.nodes[src][f'sender to {dst}'] == 'busy':
                switch_job[(src, dst)].append(busy_state-1)

            # print('*****',switch_job[(src, dst)],topology.nodes[src]['job'][(src, dst)])
            # print('#####', switch_job[(src, dst)]+ topology.nodes[src]['job'][(src, dst)])
            # if len(topology.nodes[src]['job'][(src, dst)]) > 0:
            #     # print("BBB", topology.nodes[src]['job'][(src, dst)])
            #     for B in topology.nodes[src]['job'][(src, dst)]:
            #         # print("B", B)
            #         if B['buffer'] not in WAN_buffer:
            #             switch_job[(src, dst)].append(B['buffer'])
                    # topology.nodes[src]['added_job'][(src, dst)].append(B['buffer'])
            # print('&&&&&', switch_job[(src, dst)])

        # switch_job = deduplicate_balanced(switch_job)
        switch_switch = []
        for (src, dst), value in switch_job.items():
                topology.nodes[src]['job'][(src, dst)] = []
                for buffer in value:
                    if buffer > -1:
                        topology.nodes[src]['job'][(src, dst)].append({'buffer': buffer})
                        # if buffer not in topology.nodes[src]['added_job']:
                        topology.nodes[src]['added_job'][(src, dst)].append(buffer)
                            # topology.nodes[src]['added_job'][(src, dst)].append(buffer)
                        switch_switch.append(buffer)
                # random.shuffle(topology.nodes[src]['job'][(src, dst)])

    else:
        switch_job = {}
        Switch2gpu_job = {}
        receive_buffer = topology.nodes[src]['receive_buffer']
        used_buffer = []
        for buffer in receive_buffer:
            # link = topology.edges[src, dst]
            # print("link:", link)
            buffer_index = buffer['buffer']
            if buffer['buffer'] is not None:
                for dst in successors:
                    if src in DC_0_switch and dst in DC_1_switch:
                        if buffer['buffer'] in DC1:
                            continue
                    if src in DC_1_switch and dst in DC_0_switch:
                        if buffer['buffer'] in DC0:
                            continue
                    if src in DC_0_switch and dst in DC_0_GPU:
                        if buffer['buffer'] in DC0:
                            continue
                    if src in DC_1_switch and dst in DC_1_GPU:
                        if buffer['buffer'] in DC1:
                            continue

                    if memory_state[dst][buffer_index] == 0:
                        if buffer_index not in topology.nodes[src]['added_job'][(src, dst)]:
                            if topology.nodes[dst]['type'] == 'GPU':
                                if (src, dst) not in list(Switch2gpu_job.keys()):
                                    Switch2gpu_job[(src, dst)] = []
                                    Switch2gpu_job[(src, dst)].append(buffer['buffer'])
                                    used_buffer.append(buffer)
                                else:
                                    Switch2gpu_job[(src, dst)].append(buffer['buffer'])
                                    used_buffer.append(buffer)
                                # topology.nodes[src]['job'][(src, dst)].append({'buffer': buffer['buffer']})
                                # topology.nodes[src]['added_job'][(src, dst)].append(buffer_index)
                            else:
                                if (src, dst) not in list(switch_job.keys()):
                                    switch_job[(src, dst)] = []
                                    switch_job[(src, dst)].append(buffer['buffer'])
                                    used_buffer.append(buffer)
                                else:
                                    switch_job[(src, dst)].append(buffer['buffer'])
                                    used_buffer.append(buffer)
        # print(Switch2gpu_job, switch_job, receive_buffer)
        busy_state = 0
        for (src, dst), value in switch_job.items():
            if topology.nodes[src][f'sender to {dst}'] == 'busy':
                switch_job[(src, dst)].append(busy_state - 1)
        busy_state = 0
        for (src, dst), value in Switch2gpu_job.items():
            if topology.nodes[src][f'sender to {dst}'] == 'busy':
                Switch2gpu_job[(src, dst)].append(busy_state - 1)



        # todo: add this, ltccl will be better than teccl
        # Switch2gpu_job = sort_switch2gpu_job_by_gpu_count(Switch2gpu_job, topology)
        if src in [1, 3]:
            # if random.random() < 0.3:
                switch_job = dict(reversed(list(switch_job.items())))
                # Switch2gpu_job = sort_switch2gpu_job_by_gpu_count(Switch2gpu_job, topology)
                Switch2gpu_job = dict(reversed(list(Switch2gpu_job.items())))
        # print(Switch2gpu_job)


        # todo: comment it to activate agent
        switch_job=deduplicate_balanced(switch_job)
        Switch2gpu_job=deduplicate_balanced(Switch2gpu_job)
        for (src, dst), value in switch_job.items():
            if len(topology.nodes[src]['job'][(src, dst)]) > 0:
                for B in topology.nodes[src]['job'][(src, dst)]:
                    switch_job[(src, dst)].append(B['buffer'])
        for (src, dst), value in Switch2gpu_job.items():
            if len(topology.nodes[src]['job'][(src, dst)]) > 0:
                for B in topology.nodes[src]['job'][(src, dst)]:
                    Switch2gpu_job[(src, dst)].append(B['buffer'])
        switch_switch = []
        switch_gpu = []
        for (src, dst), value in switch_job.items():
            topology.nodes[src]['job'][(src, dst)] = []
            for buffer in value:
                if buffer > -1:
                    topology.nodes[src]['job'][(src, dst)].append({'buffer': buffer})
                    topology.nodes[src]['added_job'][(src, dst)].append(buffer)
                    switch_switch.append(buffer)
        for (src, dst), value in switch_job.items():
            existing = topology.nodes[src]['added_job'][(src, dst)]
            topology.nodes[src]['added_job'][(src, dst)] = list(set(existing).union(switch_switch))
        for (src, dst), value in Switch2gpu_job.items():
            topology.nodes[src]['job'][(src, dst)] = []
            for buffer in value:
                if buffer > -1:
                    topology.nodes[src]['job'][(src, dst)].append({'buffer': buffer})
                    topology.nodes[src]['added_job'][(src, dst)].append(buffer)
                    switch_gpu.append(buffer)
            # random.shuffle(topology.nodes[src]['job'][(src, dst)])
        for buffer in used_buffer:
            # print('*****', src, buffer, topology.nodes[src]['receive_buffer'])
            if buffer in topology.nodes[src]['receive_buffer']:
                topology.nodes[src]['receive_buffer'].remove(buffer)
    # print("buffer_num_dict", buffer_num_dict)
    # print(topology.nodes[src]['job'])

    #todo: sort
    # for key, job_list in topology.nodes[src]['job'].items():
    #     # print(key, job_list)
    #     sorted_job_list = sorted(job_list, key=lambda x: buffer_num_dict[x['buffer']])
    #     topology.nodes[src]['job'][key] = sorted_job_list

        # print(key, job_list)
        # print(time, src, topology.nodes[src]['job'],Switch2gpu_job, receive_buffer)
        # for (src, dst), value in Switch2gpu_job.items():
        #     topology.nodes[src]['added_job'][(src, dst)] += switch_gpu

        #
        # for (src, dst), value in Switch2gpu_job.items():
        #     topology.nodes[src]['job'][(src, dst)].reverse()






    # print("job:", topology.nodes[src]['job'])
    """if topology.nodes[src][f'sender to {dst}'] == 'free':
        for buffer_index, buffer in memory.items():
            # print("buffer:", buffer)
            if buffer['buffer'] is not None and memory_state[dst][buffer['buffer']] == 0:
                topology.nodes[src]['job'][(src, dst)].append({'buffer': buffer['buffer']})
                # break"""

    # print(f"Time = {time}, Node = {src}, job = {topology.nodes[src]['job']}")





from copy import deepcopy


def sort_switch2gpu_job_by_gpu_count(switch2gpu_job, topology):
    """
    按 dst 节点连接的 GPU 数量降序排序 switch2gpu_job。
    保证 key:value 映射不变。
    """
    def gpu_count(dst):
        return sum(
            1 for succ in topology.successors(dst)
            if topology.nodes[succ].get("type") == "GPU"
        )

    # 保持 (key, value) 对，排序时只看 key[1] (dst) 的 GPU 数量
    sorted_items = sorted(
        switch2gpu_job.items(),
        key=lambda kv: gpu_count(kv[0][1]),
        reverse=True
    )

    # 用有序字典保证顺序
    from collections import OrderedDict
    return OrderedDict(sorted_items)

def _strip_placeholders_and_buffer_only(queue_list):
    """去除所有占位符（buffer == -1），并只保留 {'buffer': ...} 字段。"""
    cleaned = [
        item for item in queue_list
        if not (isinstance(item, dict) and item.get('buffer') == -1)
    ]
    return [
        {'buffer': (item['buffer'] if isinstance(item, dict) else item)}
        for item in cleaned
    ]

def queue(topology, memory_state, time):
    DC_0_BR = [0, 1]
    DC_1_BR = [2, 3]
    num_nodes = len(topology.nodes)
    num_GPU = num_nodes - 4
    DC_0_GPU = list(range(4, 4 + int(num_GPU / 2)))
    DC_1_GPU = list(range(4 + int(num_GPU / 2), num_GPU + 4))

    DC0_GPU_SET = set(DC_0_GPU)
    DC1_GPU_SET = set(DC_1_GPU)

    # ---- 统计各 BR 的“可用(非 busy)后继链路数”：仅统计出边且目的在对应 DC 的 GPU
    BR0_links = {}
    for BR in DC_0_BR:
        # 若是 DiGraph，用 successors；若是无向图可改 neighbors
        succ = list(topology.successors(BR))
        BR0_links[BR] = sum(
            1 for n in succ
            if n in DC0_GPU_SET and topology.nodes[BR].get(f"sender to {n}", "free") != "busy"
        )

    BR1_links = {}
    for BR in DC_1_BR:
        succ = list(topology.successors(BR))
        BR1_links[BR] = sum(
            1 for n in succ
            if n in DC1_GPU_SET and topology.nodes[BR].get(f"sender to {n}", "free") != "busy"
        )

    # -------- DC_0: GPU -> BR
    DC_0_GPU_BR = {}
    for GPU in DC_0_GPU:
        for BR in DC_0_BR:
            if (GPU, BR) in topology.nodes[GPU]['job']:
                DC_0_GPU_BR[(GPU, BR)] = []
                edge_t = topology.edges[(GPU, BR)]['transmission_latency']
                for value in topology.nodes[GPU]['job'][(GPU, BR)]:
                    DC_0_GPU_BR[(GPU, BR)].append({
                        'buffer': value['buffer'],
                        'transmission_time': edge_t
                    })
                if topology.nodes[GPU].get(f'sender to {BR}') == 'busy':
                    sent_t = topology.nodes[GPU].get(f'sender to {BR} job', {}).get('sent_time', time)
                    DC_0_GPU_BR[(GPU, BR)].append({
                        'buffer': -1,
                        'transmission_time': max(0.0, sent_t - time)
                    })

    DC_0_GPU_BR = balance_gpu_br_exact(DC_0_GPU_BR, (0, 1))
    # DC_0_GPU_BR = balance_gpu_br_cumulative(DC_0_GPU_BR, (0, 1), memory_state)

    for (GPU, BR), q in list(DC_0_GPU_BR.items()):
        topology.nodes[GPU]['job'][(GPU, BR)] = _strip_placeholders_and_buffer_only(q)

    # -------- DC_1: GPU -> BR
    DC_1_GPU_BR = {}
    for GPU in DC_1_GPU:
        for BR in DC_1_BR:
            if (GPU, BR) in topology.nodes[GPU]['job']:
                DC_1_GPU_BR[(GPU, BR)] = []
                edge_t = topology.edges[(GPU, BR)]['transmission_latency']
                for value in topology.nodes[GPU]['job'][(GPU, BR)]:
                    DC_1_GPU_BR[(GPU, BR)].append({
                        'buffer': value['buffer'],
                        'transmission_time': edge_t
                    })
                if topology.nodes[GPU].get(f'sender to {BR}') == 'busy':
                    sent_t = topology.nodes[GPU].get(f'sender to {BR} job', {}).get('sent_time', time)
                    DC_1_GPU_BR[(GPU, BR)].append({
                        'buffer': -1,
                        'transmission_time': max(0.0, sent_t - time)
                    })

    DC_1_GPU_BR = balance_gpu_br_exact(DC_1_GPU_BR, (2, 3))
    # DC_1_GPU_BR = balance_gpu_br_cumulative(DC_1_GPU_BR, (2, 3), memory_state)

    for (GPU, BR), q in list(DC_1_GPU_BR.items()):
        topology.nodes[GPU]['job'][(GPU, BR)] = _strip_placeholders_and_buffer_only(q)

    try:
        lb = memory_state.setdefault('lb_counts', defaultdict(int))
        send_log = topology.graph.get('last_tick_sends', [])
        for src, br, item in send_log:
            if isinstance(item, dict) and item.get('buffer') != -1:
                lb[(src, br)] += 1
    except Exception:
        pass




from typing import Dict, Tuple, List

def _safe_time(x) -> float:
    # 读取 item 的 transmission_time，缺失则按 0 处理
    if isinstance(x, dict):
        return float(x.get("transmission_time", 0.0))
    return float(getattr(x, "transmission_time", 0.0))

def balance_gpu_br_exact(
    dc_gpu_br: Dict[Tuple[int, int], List[dict]],
    dst_pair: Tuple[int, int] = (0, 1),
) -> Dict[Tuple[int, int], List[dict]]:
    """
    精确平衡（按 transmission_time 之和）：
    - 仅允许在同一 src 的 dst_pair 两条队列之间移动“真实任务”；
    - {'buffer': -1} 视为占位符，不可移动，保留在原 (src,dst)，并计入固定的时间和；
    - 真实任务保持原相对顺序（通过“前缀给 A，后缀给 B”的单切分实现）；
    - 目标是最小化两侧总 transmission_time 的差值（尽量接近各占一半）。
    返回新的字典，不修改传入的 dc_gpu_br。
    """
    dst_a, dst_b = dst_pair
    new_br: Dict[Tuple[int, int], List[dict]] = {}

    # -------- 1) 收集 per-src：拆分为占位符(ph)和真实任务(real)，真实任务保留完整 dict --------
    per_src = {}  # src -> {dst: {"ph": [dict...], "real": [dict...]}}
    for (src, dst), q in dc_gpu_br.items():
        if dst not in (dst_a, dst_b):
            # 非目标 dst 原样复制
            new_br[(src, dst)] = list(q)
            continue
        slot = per_src.setdefault(src, {})
        ph_list, real_list = [], []
        for item in q:
            if isinstance(item, dict) and item.get("buffer") == -1:
                ph_list.append(item)          # 占位符原样保留
            else:
                real_list.append(item)        # 真实任务（含 buffer 与 transmission_time）
        slot[dst] = {"ph": ph_list, "real": real_list}

    # -------- 2) 固定贡献（不可移动）：所有占位符的 time + 只存在一侧的真实任务 time --------
    fixed_a_time = fixed_b_time = 0.0
    flex_blocks = []   # [(src, pooled_real_items:list[dict], prefix_sums:list[float])]
    src_modes = {}     # src -> "A-only"/"B-only"/"both"

    for src, mapping in per_src.items():
        has_a = (dst_a in mapping)
        has_b = (dst_b in mapping)

        a_ph_time = sum(_safe_time(x) for x in mapping.get(dst_a, {}).get("ph", [])) if has_a else 0.0
        b_ph_time = sum(_safe_time(x) for x in mapping.get(dst_b, {}).get("ph", [])) if has_b else 0.0
        fixed_a_time += a_ph_time
        fixed_b_time += b_ph_time

        if has_a and not has_b:
            # A 侧独有：真实任务也固定
            src_modes[src] = "A-only"
            a_real = mapping[dst_a]["real"]
            fixed_a_time += sum(_safe_time(x) for x in a_real)
            new_br[(src, dst_a)] = list(mapping[dst_a]["ph"]) + list(a_real)

        elif has_b and not has_a:
            # B 侧独有
            src_modes[src] = "B-only"
            b_real = mapping[dst_b]["real"]
            fixed_b_time += sum(_safe_time(x) for x in b_real)
            new_br[(src, dst_b)] = list(mapping[dst_b]["ph"]) + list(b_real)

        elif has_a and has_b:
            # 双侧：真实任务可移动；占位符不可移动
            src_modes[src] = "both"
            pooled = list(mapping[dst_a]["real"]) + list(mapping[dst_b]["real"])  # 按原顺序拼接
            # 计算前缀和（作为“把前 k 个分给 A”的可选贡献）
            prefix = [0.0]
            acc = 0.0
            for it in pooled:
                acc += _safe_time(it)
                prefix.append(acc)  # 长度 = len(pooled)+1
            flex_blocks.append((src, pooled, prefix))

        else:
            # 两侧都没有（几乎不会发生），略过
            pass

    # -------- 3) 设定全局目标（尽量各占一半时间）并做多项选择背包 DP（以前缀和为选项） --------
    total_time = fixed_a_time + fixed_b_time + sum(prefix[-1] for _, _, prefix in flex_blocks)
    target_a = total_time / 2.0
    cap = max(0.0, target_a - fixed_a_time)  # 希望“弹性部分给 A 的时间”尽量逼近 cap

    # DP：对第 i 个弹性块，选择 k ∈ [0..len(pooled)]，贡献 prefix[k]
    # 使用浮点和，做 12 位小数取整键，避免漂移
    def key_of(x: float) -> float:
        return round(x, 12)

    # dp_map: sum_key -> 实际 sum 值
    dp_map = {key_of(0.0): 0.0}
    # 回溯表：back[i][sum_key] = (prev_sum_key, chosen_k)
    back: List[dict] = [dict()]
    back[0][key_of(0.0)] = (None, 0)

    for i, (_, _, prefix) in enumerate(flex_blocks, start=1):
        ndp = {}
        back.append({})
        for sk, sval in dp_map.items():
            for k, contrib in enumerate(prefix):  # k 从 0..L
                new_val = sval + contrib
                nk = key_of(new_val)
                # 同键冲突时，保留更接近 cap 的那个
                if (nk not in ndp) or (abs(ndp[nk] - cap) > abs(new_val - cap)):
                    ndp[nk] = new_val
                    back[i][nk] = (sk, k)
        dp_map = ndp

    # 选取最终最接近 cap 的和
    if dp_map:
        best_key = min(dp_map, key=lambda k: abs(dp_map[k] - cap))
    else:
        best_key = key_of(0.0)

    # 回溯得到每个弹性块“分给 A 的前缀长度 k”
    plan_k: List[int] = [0] * len(flex_blocks)
    cur_key = best_key
    for i in range(len(flex_blocks), 0, -1):
        prev_key, k = back[i][cur_key]
        plan_k[i - 1] = k
        cur_key = prev_key

    # -------- 4) 写回：占位符固定在原队列前面，真实任务按计划切分（保持顺序） --------
    for (idx, (src, pooled, _)) in enumerate(flex_blocks):
        k = plan_k[idx]
        a_real = pooled[:k]
        b_real = pooled[k:]
        # A 侧
        if dst_a in per_src[src]:
            a_ph = per_src[src][dst_a]["ph"]
            new_br[(src, dst_a)] = list(a_ph) + list(a_real)
        # B 侧
        if dst_b in per_src[src]:
            b_ph = per_src[src][dst_b]["ph"]
            new_br[(src, dst_b)] = list(b_ph) + list(b_real)

    # -------- 5) 补全遗漏键（例如只有占位符或前面未覆盖） --------
    for (src, dst), q in dc_gpu_br.items():
        if dst in (dst_a, dst_b) and (src, dst) not in new_br:
            # 恢复占位符+真实的原顺序（无重分配）
            mm = per_src.get(src, {}).get(dst)
            if mm is not None:
                new_br[(src, dst)] = list(mm["ph"]) + list(mm["real"])
            else:
                new_br[(src, dst)] = list(q)

    return new_br


from typing import Dict, Tuple, List


def balance_br_br_by_degree(
        dc_gpu_br: Dict[Tuple[int, int], List[dict]],
        dst_pair: Tuple[int, int],
        BR_links: Dict[int, int],
) -> Dict[Tuple[int, int], List[dict]]:
    """
    按“元素数量 / dst 出链数”做归一化平衡（仅两个 dst，例如 (0,1) 或 (2,3)）：
    - 只允许同一 src 的两条队列之间移动真实任务；
    - {'buffer': -1, ...} 视为占位符，不可移动，但计入对应 dst 的固定“数量”（=1/个）；
    - 真实任务保持原相对顺序（单切分：前缀给 dst_a，后缀给 dst_b）；
    - 目标：最小化 | (count_a / L_a) - (count_b / L_b) |。

    参数:
      dc_gpu_br: {(src, dst): [dict, ...]}  每个元素至少有 {'buffer': ...}
      dst_pair:  (dst_a, dst_b)
      BR_links:  {dst: out_degree} 每个 dst 的后继 link 数（必须 > 0）
    返回:
      新字典，元素为原 dict（占位符保留在原队列；真实任务可能在同 src 的两队列间重分配）
    """
    dst_a, dst_b = dst_pair
    if dst_a not in BR_links or dst_b not in BR_links:
        raise KeyError(f"BR_links 缺少 {dst_a} 或 {dst_b}")
    La = BR_links[dst_a]
    Lb = BR_links[dst_b]
    if La <= 0 or Lb <= 0:
        raise ValueError(f"BR_links[{dst_a}]={La}, BR_links[{dst_b}]={Lb}，必须为正整数")

    new_br: Dict[Tuple[int, int], List[dict]] = {}

    # 1) 按 src 收集：占位符(ph) 与 真实任务(real)
    per_src = {}  # src -> {dst: {"ph": [dict...], "real": [dict...]}}
    for (src, dst), q in dc_gpu_br.items():
        if dst not in (dst_a, dst_b):
            new_br[(src, dst)] = list(q)  # 非目标 dst 原样复制
            continue
        slot = per_src.setdefault(src, {})
        ph_list, real_list = [], []
        for item in q:
            if isinstance(item, dict) and item.get("buffer") == -1:
                ph_list.append(item)  # 占位符（不可移动）
            else:
                real_list.append(item)  # 真实任务（可移动）
        slot[dst] = {"ph": ph_list, "real": real_list}

    # 2) 固定计数 & 弹性块（只统计“数量”，不看时间）
    fixed_a = fixed_b = 0  # 各自不可移动元素的数量（包括占位符 + 单侧真实任务）
    flex_blocks: List[Tuple[int, List[dict]]] = []  # (src, pooled_real_items)
    for src, mapping in per_src.items():
        has_a = (dst_a in mapping)
        has_b = (dst_b in mapping)
        if has_a:
            fixed_a += len(mapping[dst_a]["ph"])
        if has_b:
            fixed_b += len(mapping[dst_b]["ph"])

        if has_a and not has_b:
            # A-only：真实任务也固定
            fixed_a += len(mapping[dst_a]["real"])
            new_br[(src, dst_a)] = list(mapping[dst_a]["ph"]) + list(mapping[dst_a]["real"])

        elif has_b and not has_a:
            # B-only
            fixed_b += len(mapping[dst_b]["real"])
            new_br[(src, dst_b)] = list(mapping[dst_b]["ph"]) + list(mapping[dst_b]["real"])

        elif has_a and has_b:
            # both：真实任务可移动
            pooled = list(mapping[dst_a]["real"]) + list(mapping[dst_b]["real"])
            flex_blocks.append((src, pooled))

        else:
            # 两侧都没有（基本不会发生）
            pass

    # 3) 推导目标：设 S = 分给 A 的弹性真实任务总数，T = 弹性真实任务总数
    #    目标最小化 | (fixed_a + S)/La - (fixed_b + (T - S))/Lb |
    #    这是 S 的线性函数，最优 S* 为:
    #    S* = [ T/Lb + fixed_b/Lb - fixed_a/La ] / (1/La + 1/Lb)
    T = sum(len(pooled) for _, pooled in flex_blocks)
    denom = (1.0 / La + 1.0 / Lb)
    S_star = (T / Lb + fixed_b / Lb - fixed_a / La) / denom if denom != 0 else T / 2.0
    # 可行域：0..T
    target_S = max(0, min(int(round(S_star)), T))

    # 4) 多项选择背包（整数 DP）：每个块 i 可贡献 k ∈ [0..len(block_i)] 到 S
    sizes = [len(pooled) for _, pooled in flex_blocks]
    dp = [False] * (T + 1)
    choose = [[-1] * (T + 1) for _ in range(len(sizes) + 1)]
    dp[0] = True

    for i, sz in enumerate(sizes, start=1):
        ndp = dp[:]  # 不能复用同一轮的更新
        for s in range(T + 1):
            if not dp[s]:
                continue
            # 选 k 个给 A（前缀长度）
            max_k = min(sz, T - s)
            for k in range(0, max_k + 1):
                if not ndp[s + k]:
                    ndp[s + k] = True
                    choose[i][s + k] = k
        dp = ndp

    # 5) 找最接近 target_S 的可达 S
    if dp[target_S]:
        best_S = target_S
    else:
        # 向两侧扩展搜索最近可达
        delta = 1
        best_S = None
        while best_S is None and (target_S - delta >= 0 or target_S + delta <= T):
            if target_S - delta >= 0 and dp[target_S - delta]:
                best_S = target_S - delta
                break
            if target_S + delta <= T and dp[target_S + delta]:
                best_S = target_S + delta
                break
            delta += 1
        if best_S is None:
            best_S = 0

    # 6) 回溯每块分给 A 的前缀数
    plan = [0] * len(sizes)
    s = best_S
    for i in range(len(sizes), 0, -1):
        k = choose[i][s]
        if k == -1:
            k = 0
        plan[i - 1] = k
        s -= k

    # 7) 写回：占位符固定在原队列前面，真实任务按计划切分（保持顺序）
    for (idx, (src, pooled)) in enumerate(flex_blocks):
        k = plan[idx]
        a_real = pooled[:k]
        b_real = pooled[k:]
        if dst_a in per_src[src]:
            a_ph = per_src[src][dst_a]["ph"]
            new_br[(src, dst_a)] = list(a_ph) + list(a_real)
        if dst_b in per_src[src]:
            b_ph = per_src[src][dst_b]["ph"]
            new_br[(src, dst_b)] = list(b_ph) + list(b_real)

    # 8) 补全遗漏键（如只有占位符或前面未覆盖）
    for (src, dst), q in dc_gpu_br.items():
        if dst in (dst_a, dst_b) and (src, dst) not in new_br:
            mm = per_src.get(src, {}).get(dst)
            if mm is not None:
                new_br[(src, dst)] = list(mm["ph"]) + list(mm["real"])
            else:
                new_br[(src, dst)] = list(q)

    return new_br


def shuffle_dict_and_values(d):
    # 1) 打乱每个 value 列表（保留原 key -> value 映射关系）
    shuffled_items = {}
    for k, v in d.items():
        v_copy = v[:]          # 浅拷贝一份
        random.shuffle(v_copy) # 打乱列表内部顺序
        shuffled_items[k] = v_copy

    # 2) 打乱 key 的顺序（只改变遍历顺序，不交换 value）
    keys = list(shuffled_items.keys())
    random.shuffle(keys)

    return {k: shuffled_items[k] for k in keys}


from typing import Dict, Tuple, List

def balance_gpu_br_sticky(
    dc_gpu_br: Dict[Tuple[int, int], List[dict]],
    dst_pair: Tuple[int, int] = (0, 1),
    topo=None,
    band: float = 0.5,      # 迟滞带：差值不超过就不调整
    step: int = 1,          # 每次最多移动的真实任务数
    br_backlog_weight: float = 1.0,  # 保留接口（主导因子是下面的自适应估计）
) -> Dict[Tuple[int, int], List[dict]]:
    """
    黏性+迟滞+小步移动的 GPU->BR 平衡器：
    - 不移动占位符（buffer == -1）；
    - 保序，只在 A/B 双端间小步移动真实任务；
    - 负载估计 = (占位符时间 + 真实任务时间) + BR 侧“等效排空时间”（自适应估计）；
    - 仅当两端负载差超过 band 才移动，每次移动不超过 step 个真实任务；
    - 记忆上次割点 topo.nodes[src][f'last_split_{A}_{B}']，增加黏性防抖。
    """
    dst_a, dst_b = dst_pair
    new_br: Dict[Tuple[int, int], List[dict]] = {}

    # 1) 按 src 聚合，拆分占位符/真实任务
    per_src: Dict[int, Dict[int, Dict[str, List[dict]]]] = {}
    for (src, dst), q in dc_gpu_br.items():
        slot = per_src.setdefault(src, {})
        ph_list, real_list = [], []
        for item in q:
            if isinstance(item, dict) and item.get("buffer") == -1:
                ph_list.append(item)
            else:
                real_list.append(item)
        slot[dst] = {"ph": ph_list, "real": real_list}

    # 2) 自适应估计 BR 侧“等效排空时间”（更温和、并行/队列可见 + EMA 平滑）
    # 只看 BR->后继 的实际发送队列与 busy，在单位时间里估计“排空成本”，
    # 再除以可用并行链路数，最后用 EMA 抑制抖动。
    br_extra = {dst_a: 0.0, dst_b: 0.0}
    if topo is not None:
        for br in (dst_a, dst_b):
            try:
                succs = list(topo.successors(br)) if hasattr(topo, 'successors') else []
                # 优先认为发往 type='switch' 是跨DC第一跳；若没有，则用全部后继
                cand = [s for s in succs if topo.nodes[s].get('type') == 'switch'] or succs

                total_work = 0.0
                available = 0
                for s in cand:
                    try:
                        tl = float(topo.edges[br, s].get('transmission_latency', 1.0))
                        q = topo.nodes[br].get('job', {}).get((br, s), [])
                        qlen = len(q)
                        busy = (topo.nodes[br].get(f'sender to {s}') == 'busy')
                        in_flight = 1 if busy else 0
                        total_work += (qlen + in_flight) * tl
                        if not busy:
                            available += 1
                    except Exception:
                        continue

                parallel = max(1, available)
                estimate = total_work / parallel

                # EMA 平滑，新值占 0.3，旧值占 0.7
                prev = float(topo.nodes[br].get('_ema_br_extra', 0.0))
                ema = 0.7 * prev + 0.3 * estimate
                topo.nodes[br]['_ema_br_extra'] = ema
                br_extra[br] = ema
            except Exception:
                br_extra[br] = 0.0

    # 3) 对每个 src 做小步调整（只动真实任务、保序）
    for src, mapping in per_src.items():
        a_ph = mapping.get(dst_a, {}).get("ph", [])
        b_ph = mapping.get(dst_b, {}).get("ph", [])
        a_real = mapping.get(dst_a, {}).get("real", [])
        b_real = mapping.get(dst_b, {}).get("real", [])
        a_ph = mapping.get(dst_a, {}).get("ph", [])
        b_ph = mapping.get(dst_b, {}).get("ph", [])
        a_real = mapping.get(dst_a, {}).get("real", [])
        b_real = mapping.get(dst_b, {}).get("real", [])
        has_a = (dst_a in mapping)
        has_b = (dst_b in mapping)
        if not (has_a and has_b):
            # 只有一侧存在时不做重分配，原样写回，避免数据丢失
            if has_a:
                new_br[(src, dst_a)] = list(a_ph) + list(a_real)
            if has_b:
                new_br[(src, dst_b)] = list(b_ph) + list(b_real)
            if topo is not None:
                key = f"last_split_{dst_a}_{dst_b}"
                topo.nodes[src][key] = len(a_real)
            continue

        def _safe_time(x):
            try:
                return float(x.get('transmission_time', 0.0))
            except Exception:
                return 0.0

        loadA = sum(_safe_time(x) for x in a_ph) + sum(_safe_time(x) for x in a_real) + br_extra[dst_a]
        loadB = sum(_safe_time(x) for x in b_ph) + sum(_safe_time(x) for x in b_real) + br_extra[dst_b]
        diff = loadA - loadB

        # 读取上次割点（默认按当前分配）
        if topo is not None:
            key = f"last_split_{dst_a}_{dst_b}"
            k_prev = topo.nodes[src].get(key, len(a_real))
        else:
            k_prev = len(a_real)

        # 迟滞 + 小步移动
        if diff > band and len(a_real) > 0:
            move = min(step, len(a_real))
            k_new = k_prev - move
        elif diff < -band and len(b_real) > 0:
            move = min(step, len(b_real))
            k_new = k_prev + move
        else:
            k_new = k_prev

        pooled = list(a_real) + list(b_real)
        k_new = max(0, min(k_new, len(pooled)))
        new_a_real = pooled[:k_new]
        new_b_real = pooled[k_new:]

        if (dst_a in mapping):
            new_br[(src, dst_a)] = list(a_ph) + new_a_real
        if (dst_b in mapping):
            new_br[(src, dst_b)] = list(b_ph) + new_b_real

        if topo is not None:
            topo.nodes[src][key] = k_new

    # 4) 保留未覆盖键
    for (src, dst), q in dc_gpu_br.items():
        if (src, dst) not in new_br:
            new_br[(src, dst)] = list(q)

    return new_br

from typing import Dict, Tuple, List, DefaultDict
from collections import defaultdict

def balance_gpu_br_cumulative(
    dc_gpu_br: Dict[Tuple[int, int], List[dict]],
    dst_pair: Tuple[int, int],
    memory_state: dict,
    *,
    keep_placeholders: bool = True,
) -> Dict[Tuple[int, int], List[dict]]:
    """
    让 GPU→BR 的 *chunk 数量* 在时间轴上尽量均衡（考虑历史累计），
    适合事件驱动模拟，避免只对当前队列做局部最优。

    原则：
      - 只移动真实任务（dict 且 buffer!=-1），保持相对顺序；占位符（buffer==-1）不动。
      - 读取 memory_state['lb_counts'] 作为 (src, br) 的历史累计计数；
        如果不存在，则创建 defaultdict(int)。
      - 对每个 src，把当前两侧真实任务合并并按“历史+当前”目标水位进行轮转分配，
        使两侧累计数量之差不超过 1。
      - 如果某个 src 只有一侧存在映射，则不做调整，原样返回，避免丢任务。
    """
    dst_a, dst_b = dst_pair

    # 历史计数容器（lazy 创建）
    # 历史计数容器（兼容 dict 或 list 类型的 memory_state；list 时回退到全局 config）
    try:
        if isinstance(memory_state, dict):
            lb: DefaultDict[Tuple[int, int], int] = memory_state.setdefault('lb_counts', defaultdict(int))
        else:
            raise TypeError
    except Exception:
        if not hasattr(config, 'lb_counts'):
            config.lb_counts = defaultdict(int)
        lb = config.lb_counts

    def is_placeholder(item: dict) -> bool:
        return isinstance(item, dict) and item.get('buffer') == -1

    # 按 src 聚合：拆分占位符/真实任务
    per_src: Dict[int, Dict[int, Dict[str, List[dict]]]] = {}
    for (src, dst), q in dc_gpu_br.items():
        bucket = per_src.setdefault(src, {})
        ph, real = [], []
        for it in q:
            (ph if is_placeholder(it) else real).append(it)
        bucket[dst] = {"ph": ph, "real": real}

    new_br: Dict[Tuple[int, int], List[dict]] = {}

    # 对每个 src 做“历史+当前”的均衡分配
    for src, mapping in per_src.items():
        has_a = dst_a in mapping
        has_b = dst_b in mapping
        if not (has_a and has_b):
            if has_a:
                new_br[(src, dst_a)] = mapping[dst_a]["ph"] + mapping[dst_a]["real"]
            if has_b:
                new_br[(src, dst_b)] = mapping[dst_b]["ph"] + mapping[dst_b]["real"]
            continue

        a_ph, b_ph = mapping[dst_a]["ph"], mapping[dst_b]["ph"]
        a_real, b_real = mapping[dst_a]["real"], mapping[dst_b]["real"]

        # 历史累计（已发/已分配）
        hist_a = int(lb[(src, dst_a)])
        hist_b = int(lb[(src, dst_b)])

        # 当前待发（仅真实任务计数）
        cur_a = len(a_real)
        cur_b = len(b_real)

        total = hist_a + hist_b + cur_a + cur_b
        target_low  = total // 2
        target_high = (total + 1) // 2  # 允许相差 <= 1

        # 合并真实任务，保持原顺序
        merged = list(a_real) + list(b_real)

        # 从历史较少的一侧开始轮转水位填充
        assign_a = hist_a
        assign_b = hist_b
        out_a: List[dict] = []
        out_b: List[dict] = []
        turn = dst_a if assign_a <= assign_b else dst_b

        for item in merged:
            if turn == dst_a:
                if assign_a < target_low or (assign_b >= target_high and assign_a < target_high):
                    out_a.append(item); assign_a += 1
                else:
                    out_b.append(item); assign_b += 1
                turn = dst_b
            else:
                if assign_b < target_low or (assign_a >= target_high and assign_b < target_high):
                    out_b.append(item); assign_b += 1
                else:
                    out_a.append(item); assign_a += 1
                turn = dst_a

        # 写回：占位符保留在前端
        new_br[(src, dst_a)] = (a_ph if keep_placeholders else []) + out_a
        new_br[(src, dst_b)] = (b_ph if keep_placeholders else []) + out_b

    # 兜底：保留未覆盖键
    for k, q in dc_gpu_br.items():
        if k not in new_br:
            new_br[k] = list(q)

    return new_br