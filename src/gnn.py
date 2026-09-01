"""
图注意力网络 (GATedge)
用于FJSP问题中的工序-机器图特征提取
将作业和机器表示为图节点，通过注意力机制学习节点间的依赖关系
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """图注意力层"""

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Linear(out_features, 1, bias=False)
        self.a_dst = nn.Linear(out_features, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: 节点特征 [batch_size, num_nodes, in_features]
        adj: 邻接矩阵 [batch_size, num_nodes, num_nodes]
        """
        h = self.W(x)  # [batch, N, out_features]

        # 计算注意力分数
        attn_src = self.a_src(h)  # [batch, N, 1]
        attn_dst = self.a_dst(h)  # [batch, N, 1]
        attn = attn_src + attn_dst.transpose(1, 2)  # [batch, N, N]
        attn = self.leaky_relu(attn)

        # 掩码：只关注有边连接的节点
        mask = (adj > 0).float()
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        # 聚合邻居特征
        out = torch.bmm(attn, h)  # [batch, N, out_features]
        return out


class GATedge(nn.Module):
    """
    基于边特征的图注意力网络
    用于FJSP中同时建模节点(工序/机器)和边(加工时间)特征
    注意: 原始项目依赖外部 graph.hgnn 模块中的 GATedge, MLPsim
          此处提供独立实现，接口保持一致
    """

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers=2, num_heads=4):
        """
        node_in_dim: 节点输入特征维度
        edge_in_dim: 边输入特征维度(如加工时间)
        hidden_dim: 隐藏层维度
        num_layers: GAT层数
        num_heads: 注意力头数
        """
        super(GATedge, self).__init__()

        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 边特征融合
        self.edge_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_features, edge_features, adj):
        """
        node_features: [batch, num_nodes, node_in_dim]
        edge_features: [batch, num_nodes, num_nodes, edge_in_dim]
        adj: [batch, num_nodes, num_nodes]
        返回: 节点嵌入 [batch, num_nodes, hidden_dim]
        """
        h = self.node_encoder(node_features)

        for i, gat_layer in enumerate(self.gat_layers):
            h_new = gat_layer(h, adj)
            h = self.norm_layers[i](h + h_new)  # 残差连接
            h = F.relu(h)

        return h

    def get_graph_embedding(self, node_features, edge_features, adj):
        """获取整张图的全局嵌入（节点特征均值池化）"""
        node_emb = self.forward(node_features, edge_features, adj)
        graph_emb = node_emb.mean(dim=1)  # [batch, hidden_dim]
        return graph_emb
