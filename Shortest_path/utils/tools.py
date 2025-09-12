import config
# from Allgather_TECCL.utils.custom import select_node_job_refactored

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
    for link_job in link_job_list:
        if time >= link_job['received_time']:
            topology.nodes[dst][f'receiver from {src}'] = 'free'
            # print(link_job)
            sent_buffer = link_job['buffer']
            # print(link_job['buffer'])
            # print((src, dst), sent_buffer, topology.nodes[dst]['memory'])
            topology.nodes[dst]['memory'][sent_buffer]['buffer'] = sent_buffer
            topology.nodes[dst]['memory'][sent_buffer]['received_time'] = time
            topology.edges[link]['job'].remove(link_job)
            if topology.edges[link]['connect'] and link_job is not None and link_job in config.connect_matrix:
                config.connect_matrix.remove(link_job)
            if topology.nodes[dst]['type'] == 'GPU':
                if link_job not in topology.nodes[dst]['receive_buffer']:

                    topology.nodes[dst]['receive_buffer'].append(link_job)
            # if topology.nodes[src]['memory'][sent_buffer]['buffer'] is None:
            #     print(
            #         f"ERROR! ! ! The node {src} want to sent buffer {sent_buffer} to node {dst}! But is None in node {src}! ")

    return


def start_receive(topology, link, time, route_table):
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
                    if topology.nodes[dst]['type'] == 'switch':
                        buffer_index = link_job['buffer']


def end_send(topology, link, time):
    src = link[0]
    dst = link[1]
    link_job_list = topology.edges[link]['job']
    max_sent_time = 0
    job = None
    for link_job in link_job_list:
        if link_job['sent_time'] > max_sent_time:
            max_sent_time = link_job['sent_time']
            job = link_job['buffer']
    if time >= max_sent_time:
        topology.nodes[src][f'sender to {dst}'] = 'free'
        """if topology.edges[link]['connect'] and job is not None and job in config.connect_matrix:
            config.connect_matrix.remove(job)"""

import random
def start_send(topology, node, time, memory_state):
    jobs_list = topology.nodes[node]['job']
    successors = list(topology.successors(node))
    for dst in successors:
        if topology.nodes[node][f'sender to {dst}'] == 'free':
            selected_job = select_node_job(topology=topology, dst=dst, time=time, node=node)
            # selected_job = select_node_job_refactored(topology, dst, time, node, config.connect_matrix)
            jobs = jobs_list[(node, dst)]
            if len(jobs) > 0:
                # job = selected_job[0]
                job = jobs[0]
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
                print(
                    f"In time {time}, node {node} sent buffer {job} to node {dst} will be received at {time + transmission_latency + propagation_latency} via {link_type}")
                memory_state[dst][job['buffer']] = 1
                # if topology.nodes[node]['memory'][job['buffer']]['buffer'] is None:
                #     print(
                #         f"ERROR! ! ! The node {node} want to sent buffer {job} to node {dst}! But is None in node {node}! ")

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


def select_node_job(topology, dst, time, node):
    predecessors = list(topology.predecessors(dst))
    jobs = {}
    estimated_time = {}
    select_buffer = {}

    for src in predecessors:
        jobs[src] = []
        job = topology.nodes[src]['job'][(src, dst)]
        estimated_time[src] = 0
        link_jobs = topology.edges[src, dst]['job']
        for link_job in link_jobs:
            if link_job["sent_time"] - time > estimated_time[src]:
                estimated_time[src] = link_job["sent_time"] - time
        for buffer in job:
            jobs[src].append(buffer['buffer'])

    for src, buffers in jobs.items():
        for buffer in buffers:
            only = True
            for checked_src, checked_buffers in jobs.items():
                if src != checked_src:
                    if buffer in checked_buffers:
                        only = False
                        if buffer not in select_buffer.keys():
                            select_buffer[buffer] = []
            if only is True:
                estimated_time[src] += topology.edges[src, dst]['transmission_latency']
            else:
                select_buffer[buffer].append(src)

    # print(jobs, select_buffer, estimated_time)
    for buffer, src_list in select_buffer.items():
        for src in src_list:
            if buffer in jobs[src]:
                """print(
                    f"For node {dst}, In node {src}, buffer {buffer} should be remove because it exists in {src}")"""
                jobs[src].remove(buffer)
    for buffer, src_list in select_buffer.items():
        src_time = {}
        for src in src_list:
            src_time[src] = estimated_time[src] + topology.edges[src, dst]['transmission_latency']

        min_src = min(src_list, key=lambda src_find: src_time[src_find])

        jobs[min_src].append(buffer)
        estimated_time[min_src] += topology.edges[min_src, dst]['transmission_latency']
    buffers = jobs[node]
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
                if {'buffer': job} in topology.nodes[src]['job'][(src, dst)]:
                    # print("remove job")
                    topology.nodes[src]['job'][(src, dst)].remove({'buffer': job})
        return [{'buffer': job}]

    # for src, job in jobs.items():


def add_node_job(topology, src, time, memory_state, sent_matrix, route_table):
    # print(time)
    memory = topology.nodes[src]['memory']
    successors = list(topology.successors(src))
    # print(memory)
    used_buffer = []
    for buffer in topology.nodes[src]['receive_buffer']:
        buffer_index = buffer['buffer']
        path = route_table[buffer_index]['policy']
        current_node_index = route_table[buffer_index]['current_node']
        current_node = path[current_node_index]
        if len(path) <= current_node_index + 1:
            continue
        if current_node != src:
            continue
        next_hop = path[current_node_index + 1]
        if (src, next_hop) in topology.nodes[src]['job'].keys():
            topology.nodes[src]['job'][(src, next_hop)].append({'buffer': buffer['buffer']})
            # route_table[buffer_index]['policy'].remove(next_hop)
            route_table[buffer_index]['current_node'] += 1
            used_buffer.append(buffer)
    # for use_buffer in used_buffer:
    #     topology.nodes[src]['receive_buffer'].remove(use_buffer)

    """print("memory:", memory)
    print("successors:", successors)
    print("job:", topology.nodes[src]['job'])"""
    # used_policy = []
    # print(topology.nodes[src]['policy'])
    # if len(topology.nodes[src]['policy']) > 0:
    #     if topology.nodes[src]['type'] == 'GPU':
    #         for P in topology.nodes[src]['policy']:
    #
    #             chunk_id = P[3]
    #             send_time = P[4]
    #             dst = P[2]
    #             dst_memory = topology.nodes[dst]['memory']
    #             if memory[chunk_id]['buffer'] is not None:
    #                 if time >= send_time or time <= send_time:
    #                     topology.nodes[src]['job'][(src, dst)].append({'buffer': chunk_id})
    #                     topology.nodes[src]['added_job'][(src, dst)].append(chunk_id)
    #                     used_policy.append(P)
    #                     # break
    #         for P in used_policy:
    #             topology.nodes[src]['policy'].remove(P)
    #     else:
    #         receive_buffer = topology.nodes[src]['receive_buffer']
    #         receive_chunk = []
    #         for buffer in receive_buffer:
    #             if buffer['buffer'] not in receive_chunk:
    #                 receive_chunk.append(buffer['buffer'])
    #         for P in topology.nodes[src]['policy']:
    #
    #             chunk_id = P[3]
    #             send_time = P[4]
    #             dst = P[2]
    #             dst_memory = topology.nodes[dst]['memory']
    #             if chunk_id in receive_chunk:
    #                 # if time >= send_time or time <= send_time:
    #                 topology.nodes[src]['job'][(src, dst)].append({'buffer': chunk_id})
    #                 topology.nodes[src]['added_job'][(src, dst)].append(chunk_id)
    #                 used_policy.append(P)
    #                 # break
    #         for P in used_policy:
    #             topology.nodes[src]['policy'].remove(P)

    # print("job:", topology.nodes[src]['job'])
    """if topology.nodes[src][f'sender to {dst}'] == 'free':
        for buffer_index, buffer in memory.items():
            # print("buffer:", buffer)
            if buffer['buffer'] is not None and memory_state[dst][buffer['buffer']] == 0:
                topology.nodes[src]['job'][(src, dst)].append({'buffer': buffer['buffer']})
                # break"""

    # print(f"Time = {time}, Node = {src}, job = {topology.nodes[src]['job']}")
# [5, 0, 4, 0, 2, 10, 2, 14, 2, 1, 8, 1, 9, 1, 3, 12, 3, 15, 11, 12, 13, 12, 3, 0, 4, 6, 7]