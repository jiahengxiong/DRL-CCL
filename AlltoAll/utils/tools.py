def check_buffer(topology):
    flag = True
    for node in topology.nodes:
        if topology.nodes[node]['type'] == 'GPU':
            buffer_list = topology.nodes[node]['memory']
            for index, buffer in buffer_list.items():
                if buffer['buffer'] is None:
                    flag = False
                    break
    return flag


def end_receive(topology, link, time):
    src = link[0]
    dst = link[1]
    link_job_list = topology.edges[link]['job']

    # 收集所有满足条件的 job
    arrived_jobs = [job for job in link_job_list if time >= job['received_time']]

    for link_job in arrived_jobs:
        link_job_list.remove(link_job)  # 现在是安全的，因为不在for link_job_list上直接迭代

        if dst == link_job['path'][-1]:
            # 放到 node dst 的 memory
            memory = topology.nodes[dst]['memory']
            debug = True
            for key, value in memory.items():
                if value['buffer'] is None:
                    topology.nodes[dst]['memory'][key] = link_job
                    debug = False
                    break
            if debug:
                print(f'bug!!! Memory is not enough!!! in node {dst}, Memory:{memory}')
        else:
            # 放到 node dst 的 received_buffer
            topology.nodes[dst]['received_buffer'].append(link_job)
    return


def start_receive(topology, link, time):
    src = link[0]
    dst = link[1]
    link_job_list = topology.edges[link]['job']
    for link_job in link_job_list:
        if time >= link_job['receive_time'] and link_job not in topology.nodes[dst][f'receiver from {src}']:
            topology.nodes[dst][f'receiver from {src}'].append(link_job)


def end_send(topology, link, time):
    src = link[0]
    dst = link[1]
    link_job_list = topology.edges[link]['job']
    for link_job in link_job_list:
        if time >= link_job['sent_time'] and link_job in topology.nodes[src][f'Sender to {dst}']:
            topology.nodes[src][f'Sender to {dst}'].remove(link_job)


def start_send(topology, node, time):
    for successor in topology.successors(node):
        if len(topology.nodes[node][f'Sender to {successor}']) == 0:
            sent_buffer = topology.nodes[node][f'sent_buffer to {successor}']
            # print(node, f'sent_buffer to {successor}', sent_buffer)
            packet = select_send_packet(sent_buffer)
            if packet is not None:
                topology.nodes[node][f'sent_buffer to {successor}'].remove(packet)
                packet['total_time'] -= topology.edges[node, successor]['total_time']
                packet['send_time'] = time
                packet['sent_time'] = time + topology.edges[node, successor]['transmission_latency']
                packet['receive_time'] = time + topology.edges[node, successor]['propagation_latency']
                packet['received_time'] = time + topology.edges[node, successor]['transmission_latency'] + \
                                          topology.edges[node, successor]['propagation_latency']
                topology.nodes[node][f'Sender to {successor}'].append(packet)
                topology.edges[node, successor]['job'].append(packet)
                link_type = topology.edges[node, successor]['type']
                print(
                    f'In time {time}, node {node} sent packet {packet["buffer"]} to node {successor} via {link_type} link, it will be '
                    f'received at {packet["received_time"]}, Sender of {node} -> {successor} will be free at '
                    f'{packet["sent_time"]}')

    return


def select_send_packet(sent_buffer):
    if len(sent_buffer) == 0:
        return None  # 如果列表为空，则返回 None
    # 利用 max 函数，比较每个字典中的 total_time
    return max(sent_buffer, key=lambda packet: packet['total_time'])


def add_node_job(topology, src, time):
    received_buffer = topology.nodes[src]['received_buffer']
    while received_buffer:
        packet = received_buffer.pop(0)  # 或者 pop()，看你要先进先出还是后进先出
        buffer = packet['buffer']
        path = packet['path']
        total_time = packet['total_time']
        next_node = path[path.index(src) + 1]
        topology.nodes[src][f'sent_buffer to {next_node}'].append(packet)


def Initialize_buffers(G, path_dict):
    node_list = list(G.nodes())
    for node in node_list:
        G.nodes[node]['received_buffer'] = []
        for successor in G.successors(node):
            G.nodes[node][f'sent_buffer to {successor}'] = []
            G.nodes[node][f'Sender to {successor}'] = []
    Initial_sent = list(path_dict.keys())
    print(Initial_sent)
    for node in node_list:
        memory = G.nodes[node]['memory']
        for key, value in memory.items():
            buffer = value['buffer']
            if buffer in Initial_sent:
                G.nodes[node]['received_buffer'].append({'buffer': buffer, 'path': path_dict[buffer]['path'],
                                                         'total_time': path_dict[buffer]['total_time']})
        print(f"Node {node} received buffer: {G.nodes[node]['received_buffer']}")
