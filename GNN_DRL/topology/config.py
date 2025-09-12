global collective
global num_chunk
global topology_name
global packet_size
global chassis
global buffer_constant
global connect_matrix
global WAN_buffer
global connectivity

# --- RL 开关与容器 ---
RL_ON = True          # 只要开了，就让 BR 上的选择走 agent
RL_DEBUG = True       # 打印 agent 的选行动/学习日志
RL_GAMMA = 0.0        # 先按 contextual bandit（收到就给奖励，不看未来）
RL_EPS = 0.3          # 探索率，先大一点，便于看到差异
RL_ALPHA = 0.2        # 学习率
RL_PENALTY_DUP = 0.5  # 重复/拥塞的惩罚系数，可再调

# registry: 每个 BR 一个 agent
AGENTS = {}           # key=(node_id), value=BRPolicyAgent
# pending: 记录从 BR 发出去但尚未到达的选择，用来在 end_receive 给奖励
PENDING = {}          # key=(src,dst,buf_id) -> dict(state, action_idx, send_time)