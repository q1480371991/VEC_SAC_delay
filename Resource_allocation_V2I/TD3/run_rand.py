# Rand (随机策略) 基准测试脚本
# 使用方式：从仓库根目录运行 python Resource_allocation_V2I/TD3/run_rand.py
# 会创建 ../log/Rand_episode{n_episode_test}_{timestamp} 用于保存结果

import os
import sys
import time
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 确保可以导入同目录的模块（当从 repo 根目录运行时）
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

import config as cfg

# 项目环境（与 main_train.py / train_td3.py 一致的导入方式）
import Environment3
from Resource_allocation_V2I.SAC import dataStruct

# -------------------------- 场景参数 --------------------------
n_veh = cfg.n_veh
BS_width = cfg.BS_width if hasattr(cfg, 'BS_width') else 1000 / 2
width = cfg.width if hasattr(cfg, 'width') else 1000
height = cfg.height if hasattr(cfg, 'height') else 1000

# 动作物理范围
max_power = cfg.max_power
min_power = cfg.min_power
max_f = cfg.max_f
min_f = cfg.min_f

# 测试参数
n_episode_test = 400  # 测试轮数，可以根据需要调整，建议与 TD3 测试轮数一致以便对比
# n_episode_test = cfg.n_episode_test # 如果 config 里有这个参数也可以用

# 创建时间戳与日志保存路径
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
# 日志路径结构保持一致：../log/算法名_episode数_时间
log_base = f'../log/Rand_numv{n_veh}_episode{n_episode_test}_{current_time}'
os.makedirs(log_base, exist_ok=True)


def save_results(name, index, E_total, reward, calculate, overload, eta1, load_rate_0, delay,elapsed_list):
    """
    保存数据并绘图，逻辑与 train_td3.py 完全一致
    """
    log_dir = f'../log/{name}_numv{n_veh}_episode{n_episode_test}_{current_time}'
    os.makedirs(log_dir, exist_ok=True)
    data = {
        'Sum_E_total': E_total,
        'Sum_reward': reward,
        'Sum_calculate': calculate,
        'Sum_overload': overload,
        'Sum_eta1': eta1,
        'Sum_load_rate_0': load_rate_0,
        'Sum_delay': delay
    }
    # 保存原始数据 pkl
    with open(f'{log_dir}/{name}_data_{index}.pkl', 'wb') as f:
        pickle.dump(data, f)

    # 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    metrics = {
        '能量消耗': E_total,
        '奖励': reward,
        '计算量': calculate,
        '过载量': overload,
        '资源浪费率': eta1,
        '卸载率': load_rate_0,
        '时延': delay,
        '耗时': elapsed_list,
        'elapsed': elapsed_list
    }
    x = np.arange(len(E_total))
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(x, values, label='Rand数据', alpha=0.6, color='gray')  # 随机策略通常用灰色表示
        window_size = 10
        # if len(values) >= window_size:
        #     smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='same')
        #     plt.plot(x, smoothed, label=f'平滑曲线（窗口={window_size}）', color='black')
        plt.xlabel('测试轮次 (Episode)')
        plt.ylabel(metric_name)
        plt.title(f'{name}算法的{metric_name}趋势')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'{log_dir}/{name}_{metric_name}.png', bbox_inches='tight')
        plt.close()


def run_rand_once(index=0):
    # 初始化环境
    up_lanes = [i / 2.0 for i in [400 + 3.5 / 2, 400 + 3.5 + 3.5 / 2, 800 + 3.5 / 2, 800 + 3.5 + 3.5 / 2]]
    down_lanes = [i / 2.0 for i in [400 - 3.5 - 3.5 / 2, 400 - 3.5 / 2, 800 - 3.5 - 3.5 / 2, 800 - 3.5 / 2]]
    left_lanes = [i / 2.0 for i in [400 + 3.5 / 2, 400 + 3.5 + 3.5 / 2, 800 + 3.5 / 2, 800 + 3.5 + 3.5 / 2]]
    right_lanes = [i / 2.0 for i in [400 - 3.5 - 3.5 / 2, 400 - 3.5 / 2, 800 - 3.5 - 3.5 / 2, 800 - 3.5 / 2]]

    env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, 0, BS_width)
    env.new_random_game()

    # 统计数据列表
    Sum_E_total_list = []
    Sum_reward_list = []
    Sum_calculate_list = []
    Sum_overload_list = []
    Sum_eta1_list = []
    Sum_load_rate_0_episode_list = []
    Sum_delay_list = []
    elapsed_list = []

    print(f"Start Rand benchmark testing for {n_episode_test} episodes...")

    for i_episode in range(n_episode_test):
        start_time = time.time()
        env.new_random_game()

        # Rand 不需要状态作为输入，但为了保持逻辑一致性，我们还是获取一下
        state = env.get_state()

        Sum_E_total_per_episode = []
        Sum_reward_per_episode = []
        Sum_calculate_per_episode = []
        Sum_overload_per_episode = []
        Sum_delay_per_episode = []
        Sum_load_rate_0_episode = []
        eta1 = []

        time_slots = dataStruct.timeSlots(start=0, end=299, slot_length=1)

        while not time_slots.is_end():
            # ================= 随机动作生成的核心逻辑 =================
            # 动作格式：[n_veh, 3] -> [Power, Frequency, Offload_Rate]
            action_all_training = np.zeros([n_veh, 3], dtype=np.float64)
            for i in range(n_veh):
                # 1. 功率 Power: 在 [min_power, max_power] 之间均匀随机
                action_all_training[i, 0] = np.random.uniform(min_power, max_power)
                # 2. 频率 Frequency: 在 [min_f, max_f] 之间均匀随机
                action_all_training[i, 1] = np.random.uniform(min_f, max_f)
                # 3. 卸载率 Offload Rate: 在 [0, 1] 之间均匀随机
                action_all_training[i, 2] = np.random.uniform(0.0, 1.0)
            # =======================================================

            # environment-specific operations（复用 main_train.py / train_td3.py 的逻辑）
            comp_n_list_true, comp_n_list = env.true_calculate_num(action_all_training)
            comp_n_list_RSU = env.calculate_num_RSU()
            env.update_buffer(comp_n_list)

            offload_num = []
            for i in range(n_veh):
                offload_num_i = int(action_all_training[i, 2] * comp_n_list_RSU)
                offload_num.append(offload_num_i)

            h_i_dB = env.overall_channel(time_slots.now())
            trans_energy_RSU = env.trans_energy_RSU(action_all_training, h_i_dB)

            # 计算 Reward 和 Metrics
            E_total, reward_tot, overload, load_rate_0, Delay_vel = env.RSU_reward1(
                action_all_training, comp_n_list_true, trans_energy_RSU, offload_num
            )

            # 记录 Step 数据
            eta1.append(overload / sum(comp_n_list) if sum(comp_n_list) != 0 else 0.0)
            if load_rate_0 == []:
                load_rate_0 = np.ones(n_veh)

            # 注意：Environment3 中 reward_tot 已经是加权的综合值，取负号或者直接使用取决于你之前的逻辑
            # train_td3.py 中有：reward = -1 * reward_tot，这里保持一致
            reward = -1 * reward_tot

            Sum_load_rate_0_episode.append(np.mean(load_rate_0))
            Sum_E_total_per_episode.append(np.sum(E_total))
            Sum_reward_per_episode.append(np.sum(reward))
            Sum_calculate_per_episode.append(np.round(np.sum(comp_n_list)))
            Sum_overload_per_episode.append(overload)
            Sum_delay_per_episode.append(sum(Delay_vel))

            # 状态更新 & 时间推进
            state_new = env.get_state()
            time_slots.add_time()

        # Episode 数据汇总
        elapsed = time.time() - start_time
        avg_reward = np.mean(Sum_reward_per_episode)
        elapsed_list.append(elapsed)

        Sum_E_total_list.append(np.mean(Sum_E_total_per_episode))
        Sum_reward_list.append(avg_reward)
        Sum_calculate_list.append(np.round(np.mean(Sum_calculate_per_episode)))
        Sum_overload_list.append(np.round(np.mean(Sum_overload_per_episode)))
        Sum_eta1_list.append(np.mean(eta1))
        Sum_load_rate_0_episode_list.append(np.mean(Sum_load_rate_0_episode))
        Sum_delay_list.append(np.mean(Sum_delay_per_episode))
        print(f'Episode {i_episode} Done. Time: {elapsed:.2f}s. Avg Reward: {np.mean(Sum_reward_per_episode):.4f}')
        print('Sum_energy_per_episode:', round(np.average(Sum_E_total_per_episode), 6))
        print('Sum_reward_per_episode:', round(np.average(Sum_reward_per_episode), 6))
        print('Sum_calculate_per_episode:', round(np.average(Sum_calculate_per_episode)))
        print('Sum_overload_rate_per_episode:', round(np.average(eta1), 6))
        print('Sum_load_rate_0_episode:', round(np.average(Sum_load_rate_0_episode), 6))
        print('Sum_delay_per_episode:', round(np.average(Sum_delay_per_episode), 6))
    # 保存模型与日志
    save_results('Rand', 0, Sum_E_total_list, Sum_reward_list, Sum_calculate_list,
                 Sum_overload_list, Sum_eta1_list, Sum_load_rate_0_episode_list, Sum_delay_list,elapsed_list)

    print(f"Rand benchmark finished. Results saved to: {log_base}")


if __name__ == "__main__":
    run_rand_once()