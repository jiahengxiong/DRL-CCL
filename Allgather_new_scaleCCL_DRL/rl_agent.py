# utils/rl_agent.py
import random
from collections import defaultdict

class BRPolicyAgent:
    """
    极简 contextual-bandit：Q(s, a) 表，ε-greedy 选动作，收到包时就学习一次。
    a 是“候选列表中的索引”，而不是 buffer id，本质上训练的是“在这个 state 下更偏好挑哪个序位的任务”。
    """
    def __init__(self, eps=0.3, alpha=0.2, gamma=0.0):
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(float)

    def _key(self, state, a):
        # state 是 tuple，a 是 int
        return (state, int(a))

    def act(self, state, num_actions):
        if num_actions <= 0:
            return None
        if random.random() < self.eps:
            return random.randrange(num_actions)
        # 贪心：挑 Q 最大的 action
        best_a, best_q = 0, float('-inf')
        for a in range(num_actions):
            q = self.Q[self._key(state, a)]
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def learn(self, state, a, reward):
        if a is None:
            return
        k = self._key(state, a)
        old = self.Q[k]
        target = reward  # contextual bandit
        self.Q[k] = old + self.alpha * (target - old)