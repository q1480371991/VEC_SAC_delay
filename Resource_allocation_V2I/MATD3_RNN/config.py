# TD3 与环境超参数（可以根据需要修改）
import numpy as np

# 场景参数（与 main_train.py 可保持一致或独立调整）
BS_width = 1000/2
width = 1000
height = 1000

# 车辆/场景数量
n_veh = 10
n_interference_vehicle = 0

# 状态 / 动作 维度（与 main_train.py 保持同义）
n_input = 2 * n_veh
n_output = 3 * n_veh

# 动作物理范围（在 train 脚本中使用）
max_power = 200
min_power = 5
max_f = 4e8
min_f = 5e7

# TD3 超参数
hidden_dim = 512
action_range = 1.0

batch_size = 256
memory_size = int(1e6)

gamma = 0.99
tau = 0.005            # target soft update; 你也可以改为 0.05 与 SAC 对齐
# actor_lr = 1e-3
# critic_lr = 1e-3
actor_lr = 1e-4
critic_lr = 1e-4


# TD3 特有参数
policy_noise = 0.2
noise_clip = 0.5
policy_delay = 2
exploration_noise = 0.1  # 采样时在 actor 输出上加入的探索噪声（高斯）


# MATD3 开关
ACTOR_SHARED = True                # 共享一个 Actor（True）或每智能体独立 Actor（False）
REWARD_MODE = 'shared_global'      # 'shared_global' 或 'shared_from_per_agent' 或 'per_agent'（向量 Critic 时）
CRITIC_VECTOR = False              # False: 标量 Q；True: 向量 Q（需搭配 REWARD_MODE='per_agent'）

# 设备
USE_GPU = True
DEVICE_IDX = 0

# 奖励权重（用于 'shared_from_per_agent' 或 'per_agent'）
REWARD_WEIGHTS = {
    'energy': 10.0,
    'delay': 4000.0,
    'overload': 1.0,   # 全局项
    'buffer': 0.01,    # 全局项
}

# RNN 后端
RNN_TYPE = 'gru'                   # 'none' | 'gru' | 'lstm'
RNN_HIDDEN_SIZE = 128