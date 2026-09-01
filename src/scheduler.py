"""
异构图神经网络调度器 (HGNNScheduler)
结合GNN和MLP，为FJSP问题生成调度决策
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gnn import GATedge
from .mlp import MLPs


class HGNNScheduler(nn.Module):
    """
    异构GNN调度器
    将FJSP问题建模为异构图：
    - 作业节点 (Job Node)
    - 机器节点 (Machine Node)
    - 工序节点 (Operation Node)
    通过GATedge提取图特征，MLP输出调度决策
    """

    def __init__(self, num_jobs, num_machines, node_feat_dim=16,
                 edge_feat_dim=1, hidden_dim=128, num_gat_layers=2):
        super(HGNNScheduler, self).__init__()

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.hidden_dim = hidden_dim

        # 节点类型嵌入
        self.job_embedding = nn.Embedding(num_jobs, node_feat_dim)
        self.machine_embedding = nn.Embedding(num_machines, node_feat_dim)

        # 异构图注意力网络
        total_nodes = num_jobs + num_machines
        self.gnn = GATedge(
            node_in_dim=node_feat_dim,
            edge_in_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers
        )

        # 决策头：选择下一个工序
        self.operation_selector = nn.Sequential(
            MLPs(hidden_dim * 2, [256, 128], 64),
            nn.Linear(64, 1)
        )

        # 机器分配头
        self.machine_assigner = nn.Sequential(
            MLPs(hidden_dim * 2, [256, 128], 64),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        """
        state: 包含当前调度状态的字典
            - job_features: [batch, num_jobs, feat_dim]
            - machine_features: [batch, num_machines, feat_dim]
            - adj: [batch, total_nodes, total_nodes]
            - edge_features: [batch, total_nodes, total_nodes, edge_dim]
            - mask: 可选动作掩码
        返回: action_probs, value
        """
        batch_size = state['job_features'].size(0)

        # 拼接作业和机器节点特征
        node_features = torch.cat([
            state['job_features'],
            state['machine_features']
        ], dim=1)

        # GNN特征提取
        node_embeddings = self.gnn(
            node_features,
            state['edge_features'],
            state['adj']
        )

        # 分离作业和机器嵌入
        job_emb = node_embeddings[:, :self.num_jobs, :]
        machine_emb = node_embeddings[:, self.num_jobs:, :]

        # 全局图嵌入
        graph_emb = node_embeddings.mean(dim=1, keepdim=True)  # [batch, 1, hidden]

        # 工序选择概率
        job_graph = torch.cat([
            job_emb,
            graph_emb.expand(-1, self.num_jobs, -1)
        ], dim=-1)
        operation_scores = self.operation_selector(job_graph).squeeze(-1)

        # 应用掩码
        if 'mask' in state:
            operation_scores = operation_scores.masked_fill(state['mask'] == 0, float('-inf'))

        operation_probs = F.softmax(operation_scores, dim=-1)

        # 机器分配概率 (基于选中的工序)
        machine_graph = torch.cat([
            machine_emb,
            graph_emb.expand(-1, self.num_machines, -1)
        ], dim=-1)
        machine_scores = self.machine_assigner(machine_graph).squeeze(-1)
        machine_probs = F.softmax(machine_scores, dim=-1)

        return operation_probs, machine_probs, graph_emb.squeeze(1)

    def get_value(self, state):
        """获取状态价值 (用于Critic)"""
        _, _, graph_emb = self.forward(state)
        value = self.value_head(graph_emb)
        return value

    def select_action(self, state, deterministic=False):
        """选择调度动作 (工序+机器)"""
        operation_probs, machine_probs, _ = self.forward(state)

        if deterministic:
            operation = torch.argmax(operation_probs, dim=-1)
            machine = torch.argmax(machine_probs, dim=-1)
        else:
            operation_dist = torch.distributions.Categorical(operation_probs)
            machine_dist = torch.distributions.Categorical(machine_probs)
            operation = operation_dist.sample()
            machine = machine_dist.sample()

        log_prob = (operation_dist.log_prob(operation) +
                    machine_dist.log_prob(machine))

        return operation, machine, log_prob
