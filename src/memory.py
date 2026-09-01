"""
经验回放内存 (Memory)
用于存储PPO训练过程中的轨迹数据
"""
import torch
import numpy as np
from collections import deque


class Memory:
    """单智能体PPO经验回放内存"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def push(self, state, action, reward, log_prob, value, done):
        """存储一步转移数据"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def get_tensors(self, device='cpu'):
        """将存储的数据转换为PyTorch张量"""
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(device)
        log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        dones = torch.FloatTensor(np.array(self.dones)).to(device)
        return states, actions, rewards, log_probs, values, dones

    def compute_returns(self, gamma=0.99, lam=0.95):
        """
        计算GAE (Generalized Advantage Estimation)
        gamma: 折扣因子
        lam: GAE的lambda参数
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage

        returns = advantages + values
        return returns, advantages

    def clear(self):
        """清空内存"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)
