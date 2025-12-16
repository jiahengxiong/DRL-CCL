import bisect
import sys
import time
import os
import config
import networkx as nx
import matplotlib.pyplot as plt
import torch



from decimal import Decimal
from Allgather_AgentinBR.Simulator.tools import add_node_job, select_node_job, start_send, combine_job, end_send, start_receive, end_receive, \
    check_buffer,queue
from Allgather_AgentinBR.utils.util import load_topology
from Allgather_AgentinBR.utils.Agent import LinkPolicyAgent
from Allgather_AgentinBR.utils.Critics import ValueCritic
from Allgather_AgentinBR.Simulator.Sim import simulation
from Allgather_AgentinBR.Simulator.PPO_trainer import PPOTrainer, PPOConfig



if __name__ == '__main__':
    num_chunk_list = [1]
    chunk_size_list = [1]
    connectivity_list = [0.3]
    collective_time = {}
    execute_time = {}

    for connectivity in connectivity_list:
        collective_time[connectivity] = {}
        execute_time[connectivity] = {}
        for num_chunk in num_chunk_list:
            collective_time[connectivity][num_chunk] = {}
            execute_time[connectivity][num_chunk] = {}
            for chunk in chunk_size_list:
                collective_time[connectivity][num_chunk][chunk] = 0
                execute_time[connectivity][num_chunk][chunk] = 0
    for connectivity in connectivity_list:
        for num_chunk in num_chunk_list:
            for chunk in chunk_size_list:
                start_time = time.time()

                # 你的代码
                config.packet_size = Decimal(str(chunk)) / Decimal(str(1024.0))
                config.num_chunk = num_chunk
                config.connectivity = connectivity
                config.chassis = 2
                config.collective = 'ALLGATHER'
                config.topology_name = 'NVD2'
                config.connect_matrix = []
                policy = []
                agent_folder = '/Users/xiongjiaheng/RDMA/CCL/Allgather_AgentinBR/Modules'
                datacenter = load_topology(packet_size=config.packet_size, num_chunk=config.num_chunk,
                                           chassis=config.chassis, name=config.topology_name)
                node_list = datacenter.topology.nodes()
                # print(f'Number of nodes: {len(node_list)}')
                # gpu_list = datacenter.gpus
                NVD2_topology = datacenter.topology

                action_set = {}
                agent_set = {}
                BR_ac = {}
                total_links = 0
                for node in node_list:
                    if NVD2_topology.nodes[node]['type'] == 'switch':
                        agent_path = agent_folder + '/' + f"BR_{node}_agent.pth"
                        L = 0
                        action_set[node] = {}
                        for neighbor in NVD2_topology.successors(node):
                            if NVD2_topology.nodes[neighbor]['type'] == 'switch':
                                action_set[node][L] = (node, neighbor)
                                L = L + 1
                                total_links = total_links + 1
                        if os.path.exists(agent_path) is False:
                            agent = LinkPolicyAgent(L)
                            torch.save(agent, agent_path)
                        agent = torch.load(agent_path, map_location="cpu", weights_only=False)
                        agent.eval()
                        agent_set[node] = agent
                        BR_ac[node] = []
                Critics_path = agent_folder + '/' + 'Critics.pth'
                if os.path.exists(Critics_path) is False:
                    critics_value = ValueCritic(total_links)
                    torch.save(critics_value, Critics_path)
                critics_value = torch.load(Critics_path, map_location="cpu", weights_only=False)

                BR_ac= simulation(collective_time,policy, agent_set, action_set)
                # print(f'BR_ac: {BR_ac}')
                config = PPOConfig(
                    gamma=0.995,
                    clip_eps=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    target_kl=0.02,
                    update_epochs=32,
                    minibatch_size=2048,
                    actor_lr=3e-4,
                    critic_lr=1e-3,
                    device="auto"
                )

                trainer = PPOTrainer(config)
                # 1) 先建优化器
                optim_actor_dict = {
                    br: torch.optim.Adam(actor.parameters(), lr=1e-5)
                    for br, actor in agent_set.items()
                }
                optim_critic = torch.optim.Adam(critics_value.parameters(), lr=1e-5)

                # 2) 确保训练模式
                for a in agent_set.values():
                    a.train()
                critics_value.train()

                # 3) 分段训练，但传入优化器以复用状态
                for ep in range(1):
                    trainer.train(
                        simulation=simulation,
                        num_iterations=1000,
                        agent_set=agent_set,
                        critics_value=critics_value,
                        collective_time=collective_time,
                        policy=[],
                        action_set=action_set,
                        verbose=True,
                        # —— 开启 gossip ——
                        enable_gossip=False,
                        gossip_every=25,  # 每 5 次 PPO 更新做一次 gossip
                        gossip_rounds=1,  # 每次 gossip 做 1 轮
                        gossip_scheme="metropolis",
                        reset_optim_after_gossip=True,  # 建议在做参数平均后清空动量
                    )
                BR_ac = simulation(collective_time, policy, agent_set, action_set)