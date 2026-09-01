# PPO算法解决FJSP问题

> 小学期项目 | 2025.07

## 项目简介

基于 PPO (Proximal Policy Optimization) 强化学习算法解决 FJSP (Flexible Job-Shop Scheduling Problem，柔性作业车间调度问题)。项目结合图神经网络 (GNN) 提取调度问题的图结构特征，通过PPO算法训练智能体学习最优调度策略，同时支持单智能体和多智能体协同调度模式。

## 技术栈

| 分类 | 技术 |
|------|------|
| 编程语言 | Python 3.8+ |
| 深度学习框架 | PyTorch |
| 强化学习算法 | PPO (近端策略优化) |
| 图神经网络 | GAT (图注意力网络) |
| 强化学习环境 | Gym |
| 数值计算 | NumPy |

## 核心模块

### 1. 数字孪生环境 (DigitalTwinEnv)
- FJSP调度问题的Gym环境封装
- 随机生成加工时间矩阵和机器可用性矩阵
- 状态空间：工序状态 + 机器负载 + 作业进度
- 动作空间：选择工序 + 分配机器
- 奖励：负完工时间 + 机器利用率 + 完成奖励

### 2. 图注意力网络 (GATedge)
- 将FJSP建模为异构图：作业节点 + 机器节点
- 多头注意力机制学习节点间依赖关系
- 边特征融合（加工时间）
- 残差连接 + LayerNorm

### 3. 异构图神经网络调度器 (HGNNScheduler)
- 结合GNN和MLP的调度决策网络
- 工序选择头 + 机器分配头
- 输出动作概率分布，支持采样和确定性决策

### 4. PPO算法
- GAE (Generalized Advantage Estimation) 优势估计
- 裁剪目标函数限制策略更新幅度
- 价值损失 + 策略损失 + 熵正则
- 梯度裁剪保证训练稳定

### 5. 多智能体PPO (MultiAgentPPO)
- CTDE (集中训练、分散执行) 框架
- 个体奖励 + 共享奖励系数 × 全局奖励
- 各智能体独立策略网络，协同优化

## 项目结构

```
ppo-fjsp-scheduler/
├── src/
│   ├── __init__.py
│   ├── env.py              # 数字孪生环境 (FJSP Gym环境)
│   ├── gnn.py              # 图注意力网络 (GATedge)
│   ├── mlp.py              # 多层感知机 (MLPs, MLPActor, MLPCritic)
│   ├── scheduler.py        # 异构GNN调度器 (HGNNScheduler)
│   ├── memory.py           # 经验回放内存 (Memory)
│   ├── ppo.py              # PPO算法实现
│   ├── multi_agent.py      # 多智能体PPO (MultiAgentMemory, MultiAgentPPO)
│   └── train.py            # 训练入口
├── requirements.txt
└── README.md
```

## 运行方式

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 单智能体训练
```bash
python -m src.train --mode single --num-jobs 10 --num-machines 5 --num-episodes 1000
```

### 3. 多智能体训练
```bash
python -m src.train --mode multi --num-agents 3 --num-jobs 10 --num-machines 5
```

### 4. 自定义参数
```bash
python -m src.train --hidden-dim 256 --lr 1e-4 --batch-size 128 --device cuda
```

## 外部依赖说明

> **注意**：原始项目中引用了外部模块 `graph.hgnn`（含 GATedge, MLPsim）和 `mlp`（含 MLPCritic, MLPActor）。本仓库已将这些模块的核心功能独立实现于 `src/gnn.py` 和 `src/mlp.py` 中，接口保持一致，可直接运行。如需使用原始外部模块，请确保相应包已安装并在Python路径中。

## 项目亮点

1. **图神经网络建模**：将FJSP调度问题建模为异构图，用GAT提取工序-机器间的结构依赖关系，比传统MLP更适合组合优化问题
2. **PPO稳定训练**：采用近端策略优化算法，通过裁剪目标函数和GAE优势估计，解决强化学习训练不稳定的问题
3. **多智能体协同**：支持多车间/多产线协同调度，CTDE框架下各智能体共享全局奖励信息
4. **数字孪生环境**：完整的Gym环境封装，支持随机问题生成，可泛化到不同规模的调度问题
5. **工程化实现**：模块化设计，环境、网络、算法、训练分离，易于扩展和复用
