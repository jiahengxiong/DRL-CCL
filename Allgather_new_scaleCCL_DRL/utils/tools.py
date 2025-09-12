import config
from Allgather_new_scaleCCL_DRL.utils.custom import select_node_job_refactored
# from config import WAN_buffer
# from generate_topo import switch_second_DC

# Decimal helper import
from decimal import Decimal

# ===== RL helpers (no external deps) =====
import random

def _ensure_rl_globals():
    # 在 config 里补齐需要的全局
    if not hasattr(config, "AGENTS"):
        config.AGENTS = {}      # {br_id: agent}
    if not hasattr(config, "PENDING"):
        config.PENDING = {}     # {(src,dst,buf): {"state":..., "action_idx":..., "send_time":...}}
    if not hasattr(config, "RL_ON"):
        config.RL_ON = True
    if not hasattr(config, "RL_DEBUG"):
        config.RL_DEBUG = False
    if not hasattr(config, "RL_EPS"):
        config.RL_EPS = 0.2     # 提高探索，便于早期学习
    if not hasattr(config, "RL_ALPHA"):
        config.RL_ALPHA = 0.5   # bandit 学习率
    if not hasattr(config, "RL_REWARD_IDEAL_LATENCY"):
        # subtract ideal (tx+prop) so reward reflects *queuing* only
        config.RL_REWARD_IDEAL_LATENCY = True
    if not hasattr(config, "BR_CONCURRENCY"):
        # 每个 BR(0/1/2/3) 允许同时在多少条出边上传输；1=严格串行，>1=部分并行
        config.BR_CONCURRENCY = 1
    if not hasattr(config, "BR_NODE_SERIAL"):
        # 若为 True：同一 BR (节点) 在所有出边上整体串行；可产生更多排队信号供 RL 学习
        config.BR_NODE_SERIAL = True
    if not hasattr(config, "BR_COOLDOWN"):
        # BR 级串行的额外冷却时间（秒），用于拉开更明显的间隔
        config.BR_COOLDOWN = 0.0
    if not hasattr(config, "RL_MIN_QUEUE_FOR_AGENT"):
        # 队列长度小于该值时不启用 agent，直接用规则；避免在无选择时引入噪声
        config.RL_MIN_QUEUE_FOR_AGENT = 2

class EpsGreedyBandit:
    """最简单 ε-greedy bandit，按 (state, action_idx) 估值"""
    def __init__(self, epsilon=0.1):
        self.eps = epsilon
        self.Q = {}  # {state: [q0,q1,...]}
    def _ensure(self, state, n):
        arr = self.Q.get(state)
        if arr is None or len(arr) != n:
            self.Q[state] = [0.0]*n
    def act(self, state, n_actions):
        self._ensure(state, n_actions)
        if n_actions == 0:
            return None
        if random.random() < self.eps:
            return random.randrange(n_actions)
        q = self.Q[state]
        m = max(q)
        cand = [i for i,v in enumerate(q) if v == m]
        return random.choice(cand)
    def learn(self, state, action_idx, reward, alpha=None):
        arr = self.Q.get(state)
        if arr is None or not (0 <= action_idx < len(arr)):
            return
        if alpha is None:
            alpha = getattr(config, "RL_ALPHA", 0.2)
        arr[action_idx] = (1-alpha)*arr[action_idx] + alpha*float(reward)

def _get_or_create_br_agent(node_id: int):
    _ensure_rl_globals()
    if node_id not in config.AGENTS:
        config.AGENTS[node_id] = EpsGreedyBandit(epsilon=config.RL_EPS)
    return config.AGENTS[node_id]

# ---- numeric helper: ensure Decimal ----
def _D(x):
    return x if isinstance(x, Decimal) else Decimal(str(x))

# 判断是否是 BR（0/1/2/3）
def _is_br(nid: int) -> bool:
    try:
        return (topology := None) is None and False  # 占位，避免引用未定义的 topology 变量
    except Exception:
        return False
# 上面这个占位防止顶层解析报错；真正使用在函数内定义 _is_br2。
# ===== end RL helpers =====



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
    link_jobs = topology.edges[link]['job']

    # 遍历拷贝，避免边遍历边删导致的问题
    for job in list(link_jobs):
        if time < job.get('received_time', float('inf')):
            continue

        # 释放接收器
        topology.nodes[dst][f'receiver from {src}'] = 'free'

        # 写入内存
        buf_id = job.get('buffer')
        if buf_id is not None:
            if 'memory' not in topology.nodes[dst]:
                topology.nodes[dst]['memory'] = {}
            if buf_id not in topology.nodes[dst]['memory']:
                topology.nodes[dst]['memory'][buf_id] = {'buffer': None, 'send_time': None, 'received_time': None}
            topology.nodes[dst]['memory'][buf_id]['buffer'] = buf_id
            topology.nodes[dst]['memory'][buf_id]['received_time'] = time

        # 跨 DC 链路释放 connect_matrix（按 connect=True 视为跨 DC）
        if topology.edges[link].get('connect', False):
            if buf_id is not None and buf_id in config.connect_matrix:
                try:
                    config.connect_matrix.remove(buf_id)
                except ValueError:
                    pass

        # === 奖励与学习（仅当 src 是 BR 且这是它的动作结果）===
        if getattr(config, "RL_ON", False) and topology.nodes[src].get('type') == 'switch' and src in (0,1,2,3) and buf_id is not None:
            k = (src, dst, buf_id)
            pend = config.PENDING.pop(k, None)
            if pend is not None:
                sent_t = float(pend.get("send_time", 0.0))
                et = float(time) - sent_t  # end-to-end measured
                ideal = 0.0
                if getattr(config, "RL_REWARD_IDEAL_LATENCY", True):
                    # subtract per-link ideal latency when available
                    try:
                        tl = float(topology.edges[link].get('transmission_latency', 0.0))
                        pl = float(topology.edges[link].get('propagation_latency', 0.0))
                        ideal = tl + pl
                    except Exception:
                        ideal = 0.0
                queuing = max(0.0, et - ideal)
                reward = -queuing  # minimize queuing delay
                agent = _get_or_create_br_agent(src)
                agent.learn(pend["state"], pend["action_idx"], reward)
                if getattr(config, "RL_DEBUG", False):
                    print(f"[RL] learn: src={src}->dst={dst}, buf={buf_id}, et={et:.6g}, ideal={ideal:.6g}, reward(=-queue)={reward:.6g}, state={pend['state']}, a={pend['action_idx']}")

        # 该 job 已完成，安全移除
        try:
            topology.edges[link]['job'].remove(job)
        except ValueError:
            pass



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
        node_is_br = (topology.nodes[node].get('type') == 'switch' and node in (0, 1, 2, 3))
        # BR 节点级串行（可选）：计算该 BR 最近一次任何出边完成发送的时间（使用 Decimal）
        serial_block = None
        if node_is_br and getattr(config, "BR_NODE_SERIAL", False):
            for _succ in topology.successors(node):
                for _ej in topology.edges[node, _succ]['job']:
                    st = _D(_ej.get('sent_time', 0))
                    if serial_block is None or st > serial_block:
                        serial_block = st
        br_cooldown = _D(getattr(config, "BR_COOLDOWN", 0.0) or 0.0)
        # BR 并发限流：同一 BR 同时最多发 K 条出边（K = config.BR_CONCURRENCY）
        if node_is_br:
            K = getattr(config, "BR_CONCURRENCY", 2)
            if K is not None and K >= 1:
                active = 0
                for _succ in topology.successors(node):
                    for _ej in topology.edges[node, _succ]['job']:
                        # 处于发送窗口内：send_time ≤ now < sent_time
                        if (_ej.get('send_time') is not None and
                            _ej.get('sent_time') is not None and
                            _ej['send_time'] <= time < _ej['sent_time']):
                            active += 1
                if active >= K:
                    continue  # 本轮不在该出边发，等下一 tick
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
                # ---- 新代码：按边序列化发送 + 记录ideal_et给RL ----
                link = topology.edges[node, dst]
                transmission_latency = _D(link['transmission_latency'])
                propagation_latency = _D(link['propagation_latency'])

                # 已有该边上的发送，找出最近一次 sent_time；新的发送不能早于它（边级）
                existing_jobs = topology.edges[node, dst]['job']
                last_sent = None
                for _j in existing_jobs:
                    st = _D(_j.get('sent_time', 0))
                    if last_sent is None or st > last_sent:
                        last_sent = st

                # 边级串行 + （可选）BR 节点级串行 + 冷却
                cands = [_D(time)]
                if last_sent is not None:
                    cands.append(last_sent)
                if serial_block is not None:
                    cands.append(serial_block + br_cooldown)
                start_at = max(cands)

                ideal_et = transmission_latency + propagation_latency  # 无排队的理想端到端

                job_rec = {
                    'buffer': job['buffer'],
                    # 做出调度决策的时间（便于统计排队）
                    'schedule_time': time,
                    # 真正发送/到达时间（考虑边上串行化）
                    'send_time': start_at,
                    'sent_time': start_at + transmission_latency,
                    'receive_time': start_at + propagation_latency,
                    'received_time': start_at + transmission_latency + propagation_latency,
                    # 方便下游奖励计算（可选）
                    'ideal_et': ideal_et,
                }

                # 入队到链路任务列表
                topology.edges[node, dst]['job'].append(job_rec)

                # 从节点待发队列移除该任务
                topology.nodes[node]['job'][(node, dst)].remove(job)

                # 标记该 sender 忙并记录当前发送的任务
                topology.nodes[node][f'sender to {dst}'] = 'busy'
                topology.nodes[node][f'sender to {dst} job'] = job_rec

                # 维护WAN_buffer等
                link_type = topology.edges[node, dst]['type']
                if topology.nodes[dst]['type'] == 'switch':
                    if job['buffer'] not in WAN_buffer:
                        WAN_buffer.append(job['buffer'])

                # 记录策略与打印
                policy.append([job['buffer'], node, dst, job_rec['received_time']])
                print(
                    f"In time {start_at}, node {node} sent buffer {{'buffer': {job['buffer']}}} to node {dst} "
                    f"will be received at {job_rec['received_time']} via {link_type}"
                )
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
    """
    非 BR(0/1/2/3 以外)：维持你原来的全局去重/估计时延逻辑。
    BR(0/1/2/3)：不做跨 (src,dst) 去重，改由本地 agent 在 (node,dst) 队列中选一个 buffer。
                 选择会写入 config.PENDING[(node,dst,buf)]，end_receive 里结算奖励。
    """
    _ensure_rl_globals()

    def _is_br2(nid: int) -> bool:
        return topology.nodes[nid].get('type') == 'switch' and nid in (0, 1, 2, 3)

    def _build_br_state_key(node, dst):
        busy = 1 if topology.nodes[node].get(f"sender to {dst}") == "busy" else 0
        q = topology.nodes[node]['job'].get((node, dst), [])
        qlen = len(q)
        qlen_bucket = 0 if qlen == 0 else (1 if qlen <= 4 else 2)
        link_t = topology.edges[node, dst]['transmission_latency']
        t_bucket = 0 if link_t < 1e-6 else (1 if link_t < 1e-5 else 2)
        return (node, dst, busy, qlen_bucket, t_bucket)

    def _enumerate_br_actions(node, dst):
        lst = topology.nodes[node]['job'].get((node, dst), [])
        seen, actions = set(), []
        for it in lst:
            if not isinstance(it, dict):
                continue
            b = it.get('buffer')
            if b is None or b in seen:
                continue
            seen.add(b)
            actions.append(b)
        return actions

    # ---------- BR：交给 agent ----------
    if _is_br2(node) and config.RL_ON:
        actions = _enumerate_br_actions(node, dst)
        if not actions:
            return []
        state = _build_br_state_key(node, dst)
        agent = _get_or_create_br_agent(node)
        # 队列长度太小时不用 agent，直接按队头；避免学习扰动拖慢基线
        if len(actions) < getattr(config, "RL_MIN_QUEUE_FOR_AGENT", 2):
            chosen_buf = actions[0]
            a_idx = 0
        else:
            a_idx = agent.act(state, len(actions))
            if a_idx is None:
                return []
            chosen_buf = actions[a_idx]
        # 记录 pending（真正发送时刻在 start_send 里覆盖）
        config.PENDING[(node, dst, chosen_buf)] = {
            "state": state,
            "action_idx": int(a_idx),
            "send_time": float(time)
        }
        if config.RL_DEBUG:
            qlen = len(topology.nodes[node]['job'].get((node, dst), []))
            print(f"[RL] BR {node}->{dst} pick buf={chosen_buf} (idx={a_idx}) state={state} qlen={qlen}")
        return [{'buffer': chosen_buf}]

    # ---------- 非 BR：保留原有逻辑 ----------
    dst_list = []
    num_nodes = len(topology.nodes)
    DC_0_GPUs = list(range(4, 4 + int((num_nodes - 4) / 2)))
    DC_1_GPUs = list(range(4 + int((num_nodes - 4) / 2), num_nodes))

    if dst == 1 or dst == 0:
        predecessors = list(topology.predecessors(0)) + list(topology.predecessors(1))
        dst_list = [0, 1]
    elif dst == 2 or dst == 3:
        predecessors = list(topology.predecessors(2)) + list(topology.predecessors(3))
        dst_list = [2, 3]
    else:
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

    jobs = {}
    estimated_time = {}
    select_buffer = {}

    for src in predecessors:
        for dst_i in dst_list:
            jobs[(src, dst_i)] = []
            if (src, dst_i) not in topology.nodes[src]['job'].keys():
                continue
            job = topology.nodes[src]['job'][(src, dst_i)]
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
                if (src, dst_i) != (checked_src, checked_dst_i) and buffer in checked_buffers:
                    only = False
                    if buffer not in select_buffer:
                        select_buffer[buffer] = []
            if only:
                estimated_time[(src, dst_i)] += topology.edges[src, dst_i]['transmission_latency']
            else:
                select_buffer[buffer].append((src, dst_i))

    for buffer, src_dst_list in select_buffer.items():
        for (src, dst_i) in src_dst_list:
            if buffer in jobs[(src, dst_i)]:
                jobs[(src, dst_i)].remove(buffer)

    for buffer, src_dst_list in select_buffer.items():
        src_time = {}
        for (src, dst_i) in src_dst_list:
            src_time[(src, dst_i)] = estimated_time[(src, dst_i)] + topology.edges[src, dst_i]['transmission_latency']
        min_src_dst_i = min(src_dst_list, key=lambda t: src_time[t])
        jobs[min_src_dst_i].append(buffer)
        estimated_time[min_src_dst_i] += topology.edges[min_src_dst_i]['transmission_latency']

    buffers = jobs.get((node, dst), [])
    if not buffers:
        return []

    if topology.edges[node, dst]['connect']:
        chosen = None
        for cand in buffers:
            if cand not in config.connect_matrix:
                config.connect_matrix.append(cand)
                chosen = cand
                break
        if chosen is None:
            return []
    else:
        chosen = buffers[0]

    for src in predecessors:
        if src == node:
            continue
        for dst_i in dst_list:
            if (src, dst_i) not in topology.nodes[src]['job']:
                continue
            if {'buffer': chosen} in topology.nodes[src]['job'][(src, dst_i)]:
                topology.nodes[src]['job'][(src, dst_i)].remove({'buffer': chosen})

    return [{'buffer': chosen}]
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
        if src in [1, 3]:
            # if random.random() < 0.3:
                switch_job = dict(reversed(list(switch_job.items())))
                Switch2gpu_job = dict(reversed(list(Switch2gpu_job.items())))
        # print(switch_job)


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





from copy import deepcopy

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
    DC_0_GPU = [4, 5, 6, 7, 8, 9]
    DC_1_GPU = [10, 11, 12, 13, 14, 15]

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

    for (GPU, BR), q in list(DC_1_GPU_BR.items()):
        topology.nodes[GPU]['job'][(GPU, BR)] = _strip_placeholders_and_buffer_only(q)

    # # -------- BR -> BR（DC_0 发往 DC_1）
    # DC_0_BR_BR = {}
    # for BR0 in DC_0_BR:
    #     for BR1 in DC_1_BR:
    #         if (BR0, BR1) in topology.nodes[BR0]['job']:
    #             orig = topology.nodes[BR0]['job'][(BR0, BR1)]
    #             DC_0_BR_BR[(BR0, BR1)] = [
    #                 (x.copy() if isinstance(x, dict) else {'buffer': x}) for x in orig
    #             ]
    #             if topology.nodes[BR0].get(f'sender to {BR1}') == 'busy':
    #                 DC_0_BR_BR[(BR0, BR1)].append({'buffer': -1})
    #
    # # —— 新增：度检查（对 (2,3) 这对 dst）——
    # L2 = BR1_links.get(2, 0)
    # L3 = BR1_links.get(3, 0)
    # if L2 > 0 and L3 > 0:
    #     DC_0_BR_BR = balance_br_br_by_degree(DC_0_BR_BR, (2, 3), BR1_links)
    # # 否则：跳过平衡，直接清理写回
    #
    # for (BR0, BR1), q in list(DC_0_BR_BR.items()):
    #     topology.nodes[BR0]['job'][(BR0, BR1)] = _strip_placeholders_and_buffer_only(q)
    #
    # # -------- BR -> BR（DC_1 发往 DC_0）
    # DC_1_BR_BR = {}
    # for BR0 in DC_1_BR:
    #     for BR1 in DC_0_BR:
    #         if (BR0, BR1) in topology.nodes[BR0]['job']:
    #             orig = topology.nodes[BR0]['job'][(BR0, BR1)]
    #             DC_1_BR_BR[(BR0, BR1)] = [
    #                 (x.copy() if isinstance(x, dict) else {'buffer': x}) for x in orig
    #             ]
    #             if topology.nodes[BR0].get(f'sender to {BR1}') == 'busy':
    #                 DC_1_BR_BR[(BR0, BR1)].append({'buffer': -1})
    #
    # # —— 新增：度检查（对 (0,1) 这对 dst）——
    # L0 = BR0_links.get(0, 0)
    # L1 = BR0_links.get(1, 0)
    # if L0 > 0 and L1 > 0:
    #     DC_1_BR_BR = balance_br_br_by_degree(DC_1_BR_BR, (0, 1), BR0_links)
    # # 否则：跳过平衡，直接清理写回
    #
    # for (BR0, BR1), q in list(DC_1_BR_BR.items()):
    #     topology.nodes[BR0]['job'][(BR0, BR1)] = _strip_placeholders_and_buffer_only(q)



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