"""
多层感知机 (MLPs)
用于PPO的Actor和Critic网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPs(nn.Module):
    """基础多层感知机"""

    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        """
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表，如 [128, 128]
        output_dim: 输出维度
        activation: 激活函数 ('relu', 'tanh', 'sigmoid')
        """
        super(MLPs, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPActor(nn.Module):
    """
    Actor网络 (策略网络)
    输出动作的概率分布
    注意: 此类依赖外部mlp模块，此处提供独立实现
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128]):
        super(MLPActor, self).__init__()
        self.backbone = MLPs(state_dim, hidden_dims, hidden_dims[-1])
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        features = self.backbone(state)
        logits = self.action_head(features)
        return F.softmax(logits, dim=-1)

    def get_action(self, state, deterministic=False):
        """根据状态选择动作"""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class MLPCritic(nn.Module):
    """
    Critic网络 (价值网络)
    输出状态价值V(s)
    注意: 此类依赖外部mlp模块，此处提供独立实现
    """

    def __init__(self, state_dim, hidden_dims=[256, 128]):
        super(MLPCritic, self).__init__()
        self.backbone = MLPs(state_dim, hidden_dims, hidden_dims[-1])
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state):
        features = self.backbone(state)
        value = self.value_head(features)
        return value.squeeze(-1)
