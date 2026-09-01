"""
多智能体PPO (Multi-Agent PPO)
用于多车间/多产线协同调度场景
每个智能体负责一个车间的调度，通过共享经验实现协同
"""
import torch
import numpy as np
from .memory import Memory
from .ppo import PPO


class MultiAgentMemory:
    """多智能体经验回放内存"""

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.agent_memories = [Memory() for _ in range(num_agents)]
        self.shared_rewards = []

    def push(self, agent_id, state, action, reward, log_prob, value, done):
        """存储单个智能体的转移数据"""
        self.agent_memories[agent_id].push(state, action, reward, log_prob, value, done)

    def push_shared_reward(self, reward):
        """存储全局共享奖励"""
        self.shared_rewards.append(reward)

    def get_agent_data(self, agent_id, device='cpu'):
        """获取指定智能体的训练数据"""
        return self.agent_memories[agent_id].get_tensors(device)

    def compute_agent_returns(self, agent_id, gamma=0.99, lam=0.95):
        """计算指定智能体的GAE"""
        return self.agent_memories[agent_id].compute_returns(gamma, lam)

    def clear(self):
        """清空所有内存"""
        for mem in self.agent_memories:
            mem.clear()
        self.shared_rewards.clear()

    def __len__(self):
        return max(len(mem) for mem in self.agent_memories)


class MultiAgentPPO:
    """
    多智能体PPO算法
    采用集中训练、分散执行 (CTDE) 框架
    - 训练时: 各智能体共享全局奖励信息
    - 执行时: 每个智能体独立决策
    """

    def __init__(self, actors, state_dim, action_dim, num_agents,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 shared_reward_coef=0.3, device='cpu'):
        """
        actors: 各智能体的策略网络列表
        num_agents: 智能体数量
        shared_reward_coef: 共享奖励系数 (个体奖励 + 系数*全局奖励)
        """
        self.num_agents = num_agents
        self.agents = []
        self.shared_reward_coef = shared_reward_coef
        self.device = device

        for i in range(num_agents):
            agent_ppo = PPO(
                actor=actors[i],
                state_dim=state_dim,
                action_dim=action_dim,
                lr=lr, gamma=gamma, lam=lam,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                device=device
            )
            self.agents.append(agent_ppo)

        self.memory = MultiAgentMemory(num_agents)

    def select_actions(self, states):
        """各智能体并行选择动作"""
        actions = []
        log_probs = []
        values = []

        for i in range(self.num_agents):
            operation, machine, log_prob, value = self.agents[i].select_action(states[i])
            actions.append((operation, machine))
            log_probs.append(log_prob)
            values.append(value)

        return actions, log_probs, values

    def store_transitions(self, states, actions, rewards, log_probs, values, dones, global_reward):
        """
        存储多智能体转移数据
        个体奖励 = 个体原始奖励 + shared_reward_coef * 全局奖励
        """
        for i in range(self.num_agents):
            combined_reward = rewards[i] + self.shared_reward_coef * global_reward
            self.memory.push(i, states[i], actions[i], combined_reward,
                           log_probs[i], values[i], dones[i])
        self.memory.push_shared_reward(global_reward)

    def update(self):
        """各智能体分别更新策略"""
        total_loss = 0.0
        for i in range(self.num_agents):
            # 将共享内存数据复制到各智能体的PPO内存中
            agent_mem = self.agents[i].memory
            src_mem = self.memory.agent_memories[i]
            agent_mem.states = src_mem.states.copy()
            agent_mem.actions = src_mem.actions.copy()
            agent_mem.rewards = src_mem.rewards.copy()
            agent_mem.log_probs = src_mem.log_probs.copy()
            agent_mem.values = src_mem.values.copy()
            agent_mem.dones = src_mem.dones.copy()

            loss = self.agents[i].update()
            total_loss += loss

        self.memory.clear()
        return total_loss / self.num_agents

    def save(self, path_prefix):
        """保存所有智能体模型"""
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent{i}.pt")

    def load(self, path_prefix):
        """加载所有智能体模型"""
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent{i}.pt")
