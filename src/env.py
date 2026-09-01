"""
数字孪生环境 (DigitalTwinEnv)
FJSP (Flexible Job-Shop Scheduling Problem) 柔性作业车间调度环境
模拟生产车间的调度过程，为PPO智能体提供交互环境
"""
import numpy as np
import gym
from gym import spaces


class DigitalTwinEnv(gym.Env):
    """
    柔性作业车间调度数字孪生环境

    状态空间:
    - 各作业的完成进度
    - 各机器的负载状态
    - 待调度工序队列

    动作空间:
    - 选择下一个要调度的工序
    - 选择执行该工序的机器

    奖励:
    - 负的最大完工时间 (Makespan)
    - 机器利用率奖励
    """

    def __init__(self, num_jobs=10, num_machines=5, num_operations_per_job=5):
        super(DigitalTwinEnv, self).__init__()

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations_per_job = num_operations_per_job
        self.total_operations = num_jobs * num_operations_per_job

        # 动作空间: (工序选择, 机器选择)
        self.action_space = spaces.MultiDiscrete([self.total_operations, self.num_machines])

        # 状态空间 (简化为向量)
        state_dim = (num_jobs * num_operations_per_job +  # 工序状态
                     num_machines +  # 机器状态
                     num_jobs)  # 作业进度
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_dim,), dtype=np.float32
        )

        # 加工时间矩阵 [job, operation, machine]
        self.processing_times = None
        # 工序是否可在某机器上加工
        self.machine_eligibility = None

        self.reset()

    def reset(self):
        """重置环境，生成新的调度问题"""
        # 随机生成加工时间 (1-100)
        self.processing_times = np.random.randint(
            1, 100, size=(self.num_jobs, self.num_operations_per_job, self.num_machines)
        )

        # 随机生成机器可用性 (每道工序至少1台机器可用)
        self.machine_eligibility = np.random.randint(
            0, 2, size=(self.num_jobs, self.num_operations_per_job, self.num_machines)
        )
        # 确保每道工序至少有一台可用机器
        for j in range(self.num_jobs):
            for o in range(self.num_operations_per_job):
                if self.machine_eligibility[j, o].sum() == 0:
                    self.machine_eligibility[j, o, np.random.randint(self.num_machines)] = 1

        # 调度状态
        self.job_current_op = np.zeros(self.num_jobs, dtype=int)  # 各作业当前工序
        self.job_completed = np.zeros(self.num_jobs, dtype=bool)  # 各作业是否完成
        self.machine_available_time = np.zeros(self.num_machines)  # 各机器可用时间
        self.machine_busy = np.zeros(self.num_machines, dtype=bool)  # 各机器是否忙碌
        self.scheduled_ops = set()  # 已调度工序集合
        self.completed_ops = set()  # 已完成工序集合
        self.current_time = 0.0
        self.makespan = 0.0

        return self._get_state()

    def step(self, action):
        """
        执行调度动作
        action: (operation_idx, machine_idx)
        """
        op_idx, machine_idx = action

        # 解析工序索引
        job_idx = op_idx // self.num_operations_per_job
        op_in_job = op_idx % self.num_operations_per_job

        # 检查动作合法性
        if not self._is_valid_action(job_idx, op_in_job, machine_idx):
            return self._get_state(), -10.0, True, {'info': 'invalid action'}

        # 获取加工时间
        proc_time = self.processing_times[job_idx, op_in_job, machine_idx]

        # 计算开始时间 (机器可用时间 和 作业前序工序完成时间 的最大值)
        job_ready_time = self._get_job_ready_time(job_idx)
        start_time = max(self.machine_available_time[machine_idx], job_ready_time)
        finish_time = start_time + proc_time

        # 更新状态
        self.machine_available_time[machine_idx] = finish_time
        self.job_current_op[job_idx] += 1
        self.scheduled_ops.add(op_idx)
        self.current_time = max(self.current_time, finish_time)

        # 检查作业是否完成
        if self.job_current_op[job_idx] >= self.num_operations_per_job:
            self.job_completed[job_idx] = True

        # 检查是否所有作业完成
        done = self.job_completed.all()
        if done:
            self.makespan = self.current_time

        # 奖励: 负的完工时间增量 + 机器利用率
        reward = self._compute_reward(start_time, finish_time, machine_idx, done)

        info = {
            'makespan': self.makespan if done else self.current_time,
            'scheduled_count': len(self.scheduled_ops)
        }

        return self._get_state(), reward, done, info

    def _is_valid_action(self, job_idx, op_in_job, machine_idx):
        """检查动作是否合法"""
        if job_idx >= self.num_jobs or op_in_job >= self.num_operations_per_job:
            return False
        if self.job_completed[job_idx]:
            return False
        if op_in_job != self.job_current_op[job_idx]:
            return False
        if not self.machine_eligibility[job_idx, op_in_job, machine_idx]:
            return False
        return True

    def _get_job_ready_time(self, job_idx):
        """获取作业的就绪时间 (前序工序完成时间)"""
        if self.job_current_op[job_idx] == 0:
            return 0.0
        # 简化: 返回当前时间 (实际应跟踪每道工序的完成时间)
        return self.current_time

    def _compute_reward(self, start_time, finish_time, machine_idx, done):
        """计算奖励"""
        # 负的加工时间 (鼓励短加工)
        reward = -(finish_time - start_time) * 0.01

        # 机器利用率奖励
        utilization = 1.0 - (start_time - self.machine_available_time[machine_idx]) / max(self.current_time, 1)
        reward += utilization * 0.1

        # 完成奖励
        if done:
            reward += 100.0 / self.makespan  # 完工时间越短奖励越高

        return reward

    def _get_state(self):
        """获取当前状态向量"""
        # 工序状态 (未调度=0, 已调度=1)
        op_state = np.zeros(self.total_operations)
        for op_idx in self.scheduled_ops:
            op_state[op_idx] = 1.0

        # 机器状态 (归一化可用时间)
        machine_state = self.machine_available_time / max(self.machine_available_time.max(), 1)

        # 作业进度
        job_progress = self.job_current_op / self.num_operations_per_job

        state = np.concatenate([op_state, machine_state, job_progress])
        return state.astype(np.float32)

    def get_action_mask(self):
        """获取合法动作掩码"""
        mask = np.zeros((self.total_operations, self.num_machines), dtype=bool)
        for job_idx in range(self.num_jobs):
            if self.job_completed[job_idx]:
                continue
            op_in_job = self.job_current_op[job_idx]
            op_idx = job_idx * self.num_operations_per_job + op_in_job
            for m in range(self.num_machines):
                if self.machine_eligibility[job_idx, op_in_job, m]:
                    mask[op_idx, m] = True
        return mask

    def render(self, mode='human'):
        """渲染当前调度状态"""
        print(f"Time: {self.current_time:.1f}, Makespan: {self.makespan:.1f}")
        print(f"Jobs completed: {self.job_completed.sum()}/{self.num_jobs}")
        print(f"Machine available times: {self.machine_available_time}")
