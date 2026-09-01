"""
PPO (Proximal Policy Optimization) 算法实现
用于训练FJSP调度策略
"""
import torch
import torch.nn as nn
import torch.optim as optim
from .memory import Memory
from .mlp import MLPCritic


class PPO:
    """
    近端策略优化算法
    核心思想: 通过裁剪目标函数限制策略更新幅度，保证训练稳定
    """

    def __init__(self, actor, state_dim, action_dim,
                 lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, k_epochs=4, batch_size=64,
                 device='cpu'):
        """
        actor: 策略网络 (HGNNScheduler)
        state_dim: 状态维度
        action_dim: 动作维度
        lr: 学习率
        gamma: 折扣因子
        lam: GAE lambda
        clip_eps: PPO裁剪范围
        value_coef: 价值损失系数
        entropy_coef: 熵正则系数
        max_grad_norm: 梯度裁剪
        k_epochs: 每次更新的迭代次数
        batch_size: 小批量大小
        """
        self.actor = actor.to(device)
        self.critic = MLPCritic(state_dim).to(device)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])

        self.memory = Memory()
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.device = device

    def select_action(self, state):
        """选择动作并存储到内存"""
        self.actor.eval()
        with torch.no_grad():
            operation, machine, log_prob = self.actor.select_action(state)
            value = self.critic(self._state_to_tensor(state))
        self.actor.train()
        return operation, machine, log_prob, value

    def store_transition(self, state, action, reward, log_prob, value, done):
        """存储转移数据"""
        self.memory.push(state, action, reward, log_prob, value, done)

    def update(self):
        """
        PPO更新
        1. 计算GAE优势函数
        2. 多次迭代小批量更新
        3. 裁剪策略比率 + 价值损失 + 熵正则
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # 获取所有数据
        states, actions, rewards, old_log_probs, values, dones = \
            self.memory.get_tensors(self.device)

        # 计算GAE
        returns, advantages = self.memory.compute_returns(self.gamma, self.lam)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        num_updates = 0

        # K轮迭代
        for _ in range(self.k_epochs):
            # 随机打乱
            indices = torch.randperm(len(states))

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # 新策略的动作概率和价值
                new_log_probs, entropy = self._evaluate_actions(batch_states, batch_actions)
                state_values = self.critic(batch_states)

                # 策略比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = nn.MSELoss()(state_values, batch_returns)

                # 总损失 = 策略损失 + 价值系数*价值损失 - 熵系数*熵
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        # 清空内存
        self.memory.clear()

        return total_loss / max(num_updates, 1)

    def _evaluate_actions(self, states, actions):
        """评估动作的对数概率和熵"""
        # 这里需要根据具体的actor接口实现
        # 简化版：假设actor输出动作分布
        operation_probs, machine_probs, _ = self.actor(states)
        operation_dist = torch.distributions.Categorical(operation_probs)
        machine_dist = torch.distributions.Categorical(machine_probs)

        log_prob = operation_dist.log_prob(actions[:, 0].long()) + \
                   machine_dist.log_prob(actions[:, 1].long())
        entropy = operation_dist.entropy().mean() + machine_dist.entropy().mean()

        return log_prob, entropy

    def _state_to_tensor(self, state):
        """将状态字典转换为张量 (简化)"""
        if isinstance(state, dict):
            return state['job_features'].mean(dim=1)  # 简化
        return torch.FloatTensor(state).to(self.device)

    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
