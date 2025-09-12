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
    src = link[0]
    dst = link[1]
    link_job_list = topology.edges[link]['job']
    flag = False
    # removed_job = []
    for link_job in link_job_list:
        if time >= link_job['received_time']:
            topology.nodes[dst][f'receiver from {src}'] = 'free'
            # print(link_job)
            sent_buffer = link_job['buffer']
            # print(link_job['buffer'])
            # print((src, dst), sent_buffer, topology.nodes[dst]['memory'])
            topology.nodes[dst]['memory'][sent_buffer]['buffer'] = sent_buffer
            topology.nodes[dst]['memory'][sent_buffer]['received_time'] = time
            # topology.nodes[dst]['receive_buffer'].remove(link_job)
            flag = True
            # if topology.nodes[dst]['type'] == 'switch':
            #     topology.nodes[dst]['buffer_limitation'] -= 1

            # topology.edges[link]['job'].remove(link_job)
            # removed_job.append(link_job)
            if topology.edges[link]['connect'] and link_job is not None and link_job in config.connect_matrix:
                config.connect_matrix.remove(link_job)
            # break
    # for link_job in removed_job:
    #     topology.edges[link]['job'].remove(link_job)

            # if topology.nodes[src]['memory'][sent_buffer]['buffer'] is None:
                # print(
                #     f"ERROR! ! ! The node {src} want to sent buffer {sent_buffer} to node {dst}! But is None in node {src}! ")

    return


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
    src = link[0]
    dst = link[1]
    link_job_list = topology.edges[link]['job']
    max_sent_time = 0
    job = None
    for link_job in link_job_list:
        if link_job['sent_time'] > max_sent_time:
            max_sent_time = link_job['sent_time']
            job = link_job['buffer']
    # if len(link_job_list) > 0:
    #     max_sent_time = link_job_list[-1]['sent_time']
    if time >= max_sent_time:
        topology.nodes[src][f'sender to {dst}'] = 'free'
        # if topology.nodes[dst]['type'] == 'switch' and topology.nodes[src]['type'] == 'GPU':
        #     topology.nodes[dst]['right'] -= 1
        # if topology.nodes[dst]['type'] == 'switch' and topology.nodes[src]['type'] == 'switch':
        #     topology.nodes[dst]['left'] -= 1
        # if job is not None and job not in WAN_buffer:
        #     WAN_buffer.append(job)
        # if topology.nodes[dst]['type'] == 'switch':
        #     topology.nodes[dst]['buffer_limitation'] -= 1
        """if topology.edges[link]['connect'] and job is not None and job in config.connect_matrix:
            config.connect_matrix.remove(job)"""

import random
def start_send(topology, node, time, memory_state, WAN_buffer, DC0, DC1,policy):
    jobs_list = topology.nodes[node]['job']
    successors = list(topology.successors(node))
    # num_nodes = len(topology.nodes)
    # DC_0_GPUs = list(range(4, 4 + int((num_nodes - 4) / 2)))
    # DC_1_GPUs = list(range(4 + int((num_nodes - 4) / 2), num_nodes))
    for dst in successors:
        # dst_out_degree = len(list(topology.successors(dst)))
        # dst_GPU = 0
        # dst_switch = 0
        # for dst_dst in topology.successors(dst):
        #     if topology.nodes[dst_dst]['type'] == 'switch':
        #         dst_switch += 1
        #     if topology.nodes[dst_dst]['type'] == 'GPU':
        #         dst_GPU += 1
        # if topology.nodes[dst]['type'] == 'switch':
        #     if topology.nodes[dst]['type'] == 'switch' and topology.nodes[node]['type'] == 'GPU':
        #         if topology.nodes[dst]['right'] >= dst_switch:
        #             continue
        #     if topology.nodes[dst]['type'] == 'switch' and topology.nodes[node]['type'] == 'switch':
        #         if topology.nodes[dst]['left'] >= dst_GPU:
        #             continue

            # if topology.nodes[dst]['buffer_limitation'] >= dst_out_degree:
            #     continue
            # if topology.nodes[dst]['buffer_limitation'] < 0:
            #     print("ERROR: topology.nodes[dst]['buffer_limitation'] <= 0")
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
                link_type = topology.edges[node, dst]['type']
                if topology.nodes[dst]['type'] == 'switch':
                    if job['buffer'] not in WAN_buffer:
                        WAN_buffer.append(job['buffer'])
                # elif dst in DC_0_GPUs:
                #     DC0.append(job['buffer'])
                # elif dst in DC_1_GPUs:
                #     DC1.append(job['buffer'])
                policy.append([job['buffer'], node, dst, time + transmission_latency + propagation_latency])
                print(
                    f"In time {time}, node {node} sent buffer {job} to node {dst} will be received at {time + transmission_latency + propagation_latency} via {link_type}")
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

    # for src, job in jobs.items():

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
        switch_job = deduplicate_balanced(switch_job)
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
        # if src in [1, 3]:
        #     # if random.random() < 0.3:
        #         switch_job = dict(reversed(list(switch_job.items())))
        #         Switch2gpu_job = dict(reversed(list(Switch2gpu_job.items())))
        # # print(switch_job)


        switch_job=deduplicate_balanced(switch_job)
        # Switch2gpu_job=deduplicate_balanced(Switch2gpu_job)
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





def queue(topology,memory_state):
    pass