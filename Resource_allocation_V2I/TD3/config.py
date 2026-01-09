# TD3 与环境超参数（可以根据需要修改）
import numpy as np

# 场景参数（与 main_train.py 可保持一致或独立调整）
BS_width = 1000/2
width = 1000
height = 1000

# 车辆/场景数量
n_veh = 5
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
