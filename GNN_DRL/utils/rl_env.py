import gym
from gym import spaces
import torch
import numpy as np


class AllGatherEnv(gym.Env):
    def __init__(self, G, packet_size_per_subchunk, GPU_list, subchunks_node, edges, outputs_to_dict, simulate_allgather_pipeline_bfs):
        super(AllGatherEnv, self).__init__()
        self.G = G
        self.packet_size_per_subchunk = packet_size_per_subchunk
        self.GPU_list = GPU_list
        self.subchunks_node = subchunks_node
        self.edges = edges
        self.outputs_to_dict = outputs_to_dict
        self.simulate_allgather_pipeline_bfs = simulate_allgather_pipeline_bfs

        # S: number of subchunks, E: number of edges
        self.S = len(subchunks_node)
        self.E = len(edges)

        # 必须用 np.float32，否则 SB3 报错
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.S, self.E), dtype=np.float32)

        # Observation space (dummy for now)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # Convert action to torch tensor
        action_tensor = torch.tensor(action, dtype=torch.float32)

        # Generate mats from action using outputs_to_dict
        mats = self.outputs_to_dict(action_tensor, self.subchunks_node, self.edges)

        # Simulate and get reward
        reward = self.simulate_allgather_pipeline_bfs(
            G=self.G,
            packet_size_per_subchunk=self.packet_size_per_subchunk,
            gpu_nodes=self.GPU_list,
            subchunk_priority_mats=mats
        )

        # Scale reward (可选，避免 reward 过大过小)
        reward = float(reward)

        # Dummy obs (和 reset 保持一致)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Gymnasium 格式需要 terminated 和 truncated
        terminated = True   # 每个 episode 一步就结束
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info