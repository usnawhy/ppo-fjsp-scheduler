"""
PPO-FJSP 训练入口
训练PPO智能体解决柔性作业车间调度问题
"""
import torch
import numpy as np
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import DigitalTwinEnv
from src.scheduler import HGNNScheduler
from src.ppo import PPO
from src.multi_agent import MultiAgentPPO


def train_single_agent(args):
    """单智能体PPO训练"""
    print("=" * 60)
    print("单智能体PPO训练 - FJSP调度问题")
    print("=" * 60)

    # 创建环境
    env = DigitalTwinEnv(
        num_jobs=args.num_jobs,
        num_machines=args.num_machines,
        num_operations_per_job=args.num_ops
    )

    # 创建策略网络
    actor = HGNNScheduler(
        num_jobs=args.num_jobs,
        num_machines=args.num_machines,
        hidden_dim=args.hidden_dim
    )

    # 创建PPO智能体
    state_dim = env.observation_space.shape[0]
    agent = PPO(
        actor=actor,
        state_dim=state_dim,
        action_dim=env.total_operations,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        k_epochs=args.k_epochs,
        batch_size=args.batch_size,
        device=args.device
    )

    # 训练循环
    best_makespan = float('inf')
    rewards_history = []

    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 选择动作
            operation, machine, log_prob, value = agent.select_action(state)
            action = (operation.item(), machine.item())

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储转移
            agent.store_transition(state, action, reward, log_prob.item(), value.item(), done)

            state = next_state
            episode_reward += reward

        # 更新策略
        loss = agent.update()
        rewards_history.append(episode_reward)

        # 记录最优解
        current_makespan = info.get('makespan', 0)
        if current_makespan < best_makespan and current_makespan > 0:
            best_makespan = current_makespan
            if args.save_model:
                os.makedirs(args.save_dir, exist_ok=True)
                agent.save(os.path.join(args.save_dir, 'best_model.pt'))

        # 打印训练信息
        if (episode + 1) % args.print_interval == 0:
            avg_reward = np.mean(rewards_history[-args.print_interval:])
            print(f"Episode [{episode+1}/{args.num_episodes}] "
                  f"Reward: {avg_reward:.2f} "
                  f"Makespan: {current_makespan:.1f} "
                  f"Best: {best_makespan:.1f} "
                  f"Loss: {loss:.4f}")

    print(f"\n训练完成! 最优Makespan: {best_makespan:.1f}")
    return best_makespan


def train_multi_agent(args):
    """多智能体PPO训练"""
    print("=" * 60)
    print("多智能体PPO训练 - 多车间协同调度")
    print("=" * 60)

    num_agents = args.num_agents
    envs = [DigitalTwinEnv(args.num_jobs, args.num_machines, args.num_ops)
            for _ in range(num_agents)]

    actors = [HGNNScheduler(args.num_jobs, args.num_machines, hidden_dim=args.hidden_dim)
              for _ in range(num_agents)]

    state_dim = envs[0].observation_space.shape[0]
    multi_agent = MultiAgentPPO(
        actors=actors,
        state_dim=state_dim,
        action_dim=envs[0].total_operations,
        num_agents=num_agents,
        lr=args.lr,
        device=args.device
    )

    for episode in range(args.num_episodes):
        states = [env.reset() for env in envs]
        dones = [False] * num_agents
        episode_rewards = [0] * num_agents

        while not all(dones):
            actions, log_probs, values = multi_agent.select_actions(states)

            next_states = []
            rewards = []
            new_dones = []
            for i in range(num_agents):
                if not dones[i]:
                    action = (actions[i][0].item(), actions[i][1].item())
                    next_state, reward, done, info = envs[i].step(action)
                    next_states.append(next_state)
                    rewards.append(reward)
                    new_dones.append(done)
                    episode_rewards[i] += reward
                else:
                    next_states.append(states[i])
                    rewards.append(0)
                    new_dones.append(True)

            global_reward = sum(rewards) / num_agents
            multi_agent.store_transitions(
                states, actions, rewards,
                [lp.item() for lp in log_probs],
                [v.item() for v in values],
                new_dones, global_reward
            )
            states = next_states
            dones = new_dones

        loss = multi_agent.update()

        if (episode + 1) % args.print_interval == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"Episode [{episode+1}/{args.num_episodes}] "
                  f"Avg Reward: {avg_reward:.2f} Loss: {loss:.4f}")

    print("多智能体训练完成!")


def main():
    parser = argparse.ArgumentParser(description='PPO-FJSP 调度问题训练')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'],
                        help='训练模式: single(单智能体) / multi(多智能体)')
    parser.add_argument('--num-jobs', type=int, default=10, help='作业数量')
    parser.add_argument('--num-machines', type=int, default=5, help='机器数量')
    parser.add_argument('--num-ops', type=int, default=5, help='每个作业的工序数')
    parser.add_argument('--num-agents', type=int, default=3, help='多智能体数量')
    parser.add_argument('--hidden-dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--clip-eps', type=float, default=0.2, help='PPO裁剪范围')
    parser.add_argument('--k-epochs', type=int, default=4, help='每次更新迭代次数')
    parser.add_argument('--batch-size', type=int, default=64, help='小批量大小')
    parser.add_argument('--num-episodes', type=int, default=1000, help='训练回合数')
    parser.add_argument('--print-interval', type=int, default=10, help='打印间隔')
    parser.add_argument('--save-model', action='store_true', help='是否保存模型')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cpu', help='训练设备: cpu / cuda')

    args = parser.parse_args()

    if args.mode == 'single':
        train_single_agent(args)
    else:
        train_multi_agent(args)


if __name__ == '__main__':
    main()
