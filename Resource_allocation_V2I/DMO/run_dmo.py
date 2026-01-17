# DMO 运行入口脚本
# 运行方式：从仓库根目录运行 python Resource_allocation_V2I/DMO/run_dmo.py (假设你放到了 DMO 文件夹)

import os
import sys
import time
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 路径处理
this_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 run_dmo.py 和 train_td3.py 在同一级，或者你需要调整 sys.path
root_dir = os.path.dirname(this_dir)  # 根据你的实际目录结构调整
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 引入 DMO 求解器
from dmo_solver import DMOSolver
import config as cfg

# 项目环境
import Environment3
from Resource_allocation_V2I.SAC import dataStruct

# --- 参数设置 ---
n_veh = cfg.n_veh
n_input = cfg.n_input
n_output = cfg.n_output

BS_width = cfg.BS_width if hasattr(cfg, 'BS_width') else 1000 / 2
width = cfg.width if hasattr(cfg, 'width') else 1000
height = cfg.height if hasattr(cfg, 'height') else 1000

# 动作物理范围
max_power = cfg.max_power
min_power = cfg.min_power
max_f = cfg.max_f
min_f = cfg.min_f

# DMO 参数
n_step_per_episode = 100  # 也可以设为 300 (time_slots)
n_episode_test = 5  # DMO 比较慢，测试轮数可以少一点，或者保持一致

# 创建日志路径 (沿用现有结构)
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
# 注意：这里名字改为 DMO
log_base = f'../log/DMO_numv{n_veh}_episode{n_episode_test}_{current_time}'
os.makedirs(log_base, exist_ok=True)

# 构造动作边界 (n_veh * 3)
# 顺序: [Power, Frequency, Ratio]
action_bounds = []
for _ in range(n_veh):
    action_bounds.append((min_power, max_power))  # Power
    action_bounds.append((min_f, max_f))  # Frequency
    action_bounds.append((0.0, 1.0))  # Ratio

# 初始化 DMO 求解器
dmo_solver = DMOSolver(
    n_veh=n_veh,
    n_dim=n_veh * 3,
    action_bounds=action_bounds,
    pop_size=20,  # 种群大小，越大越准但越慢
    max_iter=30  # 迭代次数
)


# 复用你的 save_results 函数 (完全复制过来)
def save_results(name, index, E_total, reward, calculate, overload, eta1, load_rate_0, delay ,elapsed_list):
    log_dir = f'../log/{name}_numv{n_veh}_episode{n_episode_test}_{current_time}'
    os.makedirs(log_dir, exist_ok=True)
    data = {
        'Sum_E_total': E_total,
        'Sum_reward': reward,
        'Sum_calculate': calculate,
        'Sum_overload': overload,
        'Sum_eta1': eta1,
        'Sum_load_rate_0': load_rate_0,
        'Sum_delay': delay,
        'elapsed':elapsed_list
    }
    with open(f'{log_dir}/{name}_data_{index}.pkl', 'wb') as f:
        pickle.dump(data, f)

    # 绘图部分 (保持不变)
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
        '耗时':elapsed_list
    }
    x = np.arange(len(E_total))
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(x, values, label='DMO数据', alpha=0.6)  # Label 改一下
        window_size = 10
        if len(values) >= window_size:
            smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='same')
            plt.plot(x, smoothed, label=f'平滑曲线（窗口={window_size}）', color='red')
        plt.xlabel('训练轮次 (Episode)')
        plt.ylabel(metric_name)
        plt.title(f'{name}算法的{metric_name}趋势')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'{log_dir}/{name}_{metric_name}.png', bbox_inches='tight')
        plt.close()


def run_dmo_once(index=0):
    # 环境构造
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
    elapsed_list=[]

    for i_episode in range(n_episode_test):
        start_time = time.time()
        print(f'------ DMO/Episode {i_episode} ------')
        env.new_random_game()

        # DMO 不需要 state 来做推理，但需要 state 来更新环境
        state = env.get_state()

        Sum_E_total_per_episode = []
        Sum_reward_per_episode = []
        Sum_calculate_per_episode = []
        Sum_overload_per_episode = []
        Sum_delay_per_episode = []
        Sum_load_rate_0_episode = []
        eta1 = []

        time_slots = dataStruct.timeSlots(start=0, end=299, slot_length=1)  # 假设只跑部分 slot 验证速度

        # 为了演示速度，这里可以限制 slot 数量，正式跑可以使用完整 time_slots
        step_count = 0
        while not time_slots.is_end():
            # 使用 DMO 求解当前时刻的最佳动作
            # 注意：solve 会进行多次迭代，计算比较耗时
            best_action_flat = dmo_solver.solve(env, state)

            # 将 flat 动作转换为矩阵形式
            action_matrix = np.zeros((n_veh, 3))
            for k in range(n_veh):
                idx = k * 3
                action_matrix[k, 0] = best_action_flat[idx + 0]
                action_matrix[k, 1] = best_action_flat[idx + 1]
                action_matrix[k, 2] = best_action_flat[idx + 2]

            # --- 以下逻辑与 train_td3 保持一致 ---

            # 环境交互
            comp_n_list_true, comp_n_list = env.true_calculate_num(action_matrix)
            comp_n_list_RSU = env.calculate_num_RSU()
            env.update_buffer(comp_n_list)

            offload_num = []
            for k in range(n_veh):
                offload_num_i = int(action_matrix[k, 2] * comp_n_list_RSU)
                offload_num.append(offload_num_i)

            h_i_dB = env.overall_channel(time_slots.now())
            trans_energy_RSU = env.trans_energy_RSU(action_matrix, h_i_dB)

            # 获取 Reward 和 Metrics
            E_total, reward_tot, overload, load_rate_0, Delay_vel = env.RSU_reward1(
                action_matrix, comp_n_list_true, trans_energy_RSU, offload_num
            )

            # 记录数据
            eta1.append(overload / sum(comp_n_list) if sum(comp_n_list) != 0 else 0.0)
            if load_rate_0 == []:
                load_rate_0 = np.ones(n_veh)

            reward = reward_tot  # DMO 已经是最大化 reward
            E_total = 1 * E_total

            Sum_load_rate_0_episode.append(np.mean(load_rate_0))
            Sum_E_total_per_episode.append(np.sum(E_total))
            Sum_reward_per_episode.append(-np.sum(reward))
            Sum_calculate_per_episode.append(np.round(np.sum(comp_n_list)))
            Sum_overload_per_episode.append(overload)
            Sum_delay_per_episode.append(sum(Delay_vel))

            # 状态更新 (DMO本身不需要 next_state 训练，但环境需要推进)
            state_new = env.get_state()
            state = state_new
            time_slots.add_time()

            step_count += 1
            # 如果太慢，可以加上 if step_count > 50: break

        # Episode 统计
        Sum_E_total_list.append(np.mean(Sum_E_total_per_episode))
        Sum_reward_list.append(np.mean(Sum_reward_per_episode))
        Sum_calculate_list.append(np.round(np.mean(Sum_calculate_per_episode)))
        Sum_overload_list.append(np.round(np.mean(Sum_overload_per_episode)))
        Sum_eta1_list.append(np.mean(eta1))
        Sum_load_rate_0_episode_list.append(np.mean(Sum_load_rate_0_episode))
        Sum_delay_list.append(np.mean(Sum_delay_per_episode))

        elapsed = time.time() - start_time
        elapsed_list.append(elapsed)
        print(f'Episode {i_episode} Done. Time: {elapsed:.2f}s. Avg Reward: {np.mean(Sum_reward_per_episode):.4f}')
        print('Sum_energy_per_episode:', round(np.average(Sum_E_total_per_episode), 6))
        print('Sum_reward_per_episode:', round(np.average(Sum_reward_per_episode), 6))
        print('Sum_calculate_per_episode:', round(np.average(Sum_calculate_per_episode)))
        print('Sum_overload_rate_per_episode:', round(np.average(eta1), 6))
        print('Sum_load_rate_0_episode:', round(np.average(Sum_load_rate_0_episode), 6))
        print('Sum_delay_per_episode:', round(np.average(Sum_delay_per_episode), 6))

    # 保存结果
    save_results('DMO', 0, Sum_E_total_list, Sum_reward_list, Sum_calculate_list,
                 Sum_overload_list, Sum_eta1_list, Sum_load_rate_0_episode_list, Sum_delay_list,elapsed_list)


if __name__ == "__main__":
    run_dmo_once()
    print("DMO simulation finished.")