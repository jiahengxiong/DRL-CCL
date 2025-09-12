from collections import defaultdict
from decimal import Decimal
# Allgather/utils/custom.py
from collections import defaultdict
import random

BR_IDS = {0, 1, 2, 3}

_BR_AGENT_REGISTRY = {}  # node_id -> BRAgent

def get_br_agent(node_id: int):
    """拿/建 BR 的 agent。非 BR 直接返回 None。"""
    if node_id not in BR_IDS:
        return None
    if node_id not in _BR_AGENT_REGISTRY:
        _BR_AGENT_REGISTRY[node_id] = BRAgent(node_id)
    return _BR_AGENT_REGISTRY[node_id]


def _bin_int(x: int, bins=(0, 1, 2, 4, 8, 16, 9999999)):
    for i, b in enumerate(bins):
        if x <= b:
            return i
    return len(bins)


def build_br_state_key(topology, node: int, dst: int):
    """
    用很便宜的特征构建一个 state key：
      - 当前 link 是否 busy
      - 该 (node,dst) 队列长度粗分箱
      - 该 dst 的其他出边有多少空闲（帮助 agent 感知拥塞）
    """
    # link busy
    busy = 1 if topology.nodes[node].get(f"sender to {dst}", "free") == "busy" else 0

    # 当前 (node,dst) 队列长度
    q = topology.nodes[node]["job"].get((node, dst), [])
    qlen_bin = _bin_int(len(q))

    # 这个 dst 的其他进入队列长度汇总（帮助反映下游拥塞）
    # 这里用下游节点的“可用 GPU 链路数量”近似，而不是扫全图
    succ = list(topology.successors(dst))
    free_ports = 0
    for n in succ:
        if topology.nodes[dst].get(f"sender to {n}", "free") != "busy":
            free_ports += 1
    free_ports_bin = _bin_int(free_ports, bins=(0, 1, 2, 3, 4, 8, 9999999))

    return (busy, qlen_bin, free_ports_bin)


def enumerate_br_actions(topology, node: int, dst: int):
    """
    枚举此刻 (node,dst) 可发送的 buffer 列表。
    仅在 node 是 BR 且链路空闲时返回候选。
    """
    if topology.nodes[node].get(f"sender to {dst}", "free") == "busy":
        return []
    job_list = topology.nodes[node]["job"].get((node, dst), [])
    return [item["buffer"] for item in job_list if isinstance(item, dict) and item.get("buffer") is not None]


class BRAgent:
    """
    非常轻量的 Q-learning（按 buffer 作为 action）。
    状态离散化为 build_br_state_key 输出的三元组。
    """

    def __init__(self, node_id: int, alpha=0.2, gamma=0.9, eps=0.1):
        self.node_id = node_id
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        # Q[(state_key)][buffer] = value
        self.Q = defaultdict(lambda: defaultdict(float))
        self.last = None  # (state_key, buffer)

    def act(self, state_key, actions):
        """
        ε-greedy 选择一个 buffer；如果没有可选，返回 None。
        """
        if not actions:
            self.last = None
            return None

        # 初始化所有动作
        _ = [self.Q[state_key][a] for a in actions]

        if random.random() < self.eps:
            a = random.choice(actions)
        else:
            best = None
            best_q = -1e30
            for a in actions:
                q = self.Q[state_key][a]
                if q > best_q:
                    best_q = q
                    best = a
            a = best

        self.last = (state_key, a)
        return a

    def learn(self, reward, next_state=None, next_actions=None):
        """
        简单的 TD(0)。next_state/next_actions 目前可不传，方便以后扩展。
        """
        if self.last is None:
            return
        skey, buf = self.last
        old_q = self.Q[skey][buf]

        max_next = 0.0
        if next_state is not None and next_actions:
            # 估值下一步
            # 预热
            _ = [self.Q[next_state][a] for a in next_actions]
            max_next = max(self.Q[next_state][a] for a in next_actions) if next_actions else 0.0

        target = reward + self.gamma * max_next
        self.Q[skey][buf] = old_q + self.alpha * (target - old_q)
        self.last = None

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
