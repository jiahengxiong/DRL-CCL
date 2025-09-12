from collections import defaultdict
from decimal import Decimal


def estimate_available_time(link_jobs, current_time):
    future_jobs = [j['sent_time'] for j in link_jobs if j['sent_time'] > current_time]
    if future_jobs:
        return max(future_jobs) - current_time
    return Decimal('0')


def collect_jobs_and_estimates(topology, dst, current_time):
    predecessors = list(topology.predecessors(dst))
    jobs = {}
    estimated_time = {}

    for src in predecessors:
        job_list = topology.nodes[src]['job'][(src, dst)]
        link_jobs = topology.edges[src, dst]['job']
        jobs[src] = [buffer['buffer'] for buffer in job_list]
        estimated_time[src] = estimate_available_time(link_jobs, current_time)

    return jobs, estimated_time, predecessors


def resolve_buffer_conflicts(jobs):
    buffer_owners = defaultdict(list)
    for src, buffers in jobs.items():
        for buffer in buffers:
            buffer_owners[buffer].append(src)
    return buffer_owners


def assign_unique_buffers(jobs, estimated_time, topology, dst):
    buffer_owners = resolve_buffer_conflicts(jobs)
    for buffer, src_list in buffer_owners.items():
        if len(src_list) == 1:
            src = src_list[0]
            estimated_time[src] += topology.edges[src, dst]['transmission_latency']
        else:
            best_src = min(
                src_list,
                key=lambda s: estimated_time[s] + topology.edges[s, dst]['transmission_latency']
            )
            for src in src_list:
                if buffer in jobs[src] and src != best_src:
                    jobs[src].remove(buffer)
            if buffer not in jobs[best_src]:
                jobs[best_src].append(buffer)
                estimated_time[best_src] += topology.edges[best_src, dst]['transmission_latency']
    return jobs


def select_node_job_refactored(topology, dst, time, node, connect_matrix):
    jobs, estimated_time, predecessors = collect_jobs_and_estimates(topology, dst, time)
    jobs = assign_unique_buffers(jobs, estimated_time, topology, dst)

    buffers = jobs.get(node, [])
    if not buffers:
        return []

    selected_job = None
    if topology.edges[node, dst]['connect']:
        for job in buffers:
            if job not in connect_matrix:
                connect_matrix.append(job)
                selected_job = job
                break
    else:
        selected_job = buffers[0]

    if selected_job is None:
        return []

    for src in predecessors:
        if src != node:
            src_jobs = topology.nodes[src]['job'][(src, dst)]
            for item in src_jobs:
                if item['buffer'] == selected_job:
                    topology.nodes[src]['job'][(src, dst)].remove(item)
                    break

    return [{'buffer': selected_job}]
