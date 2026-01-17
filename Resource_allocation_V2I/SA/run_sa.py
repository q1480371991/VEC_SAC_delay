# 模拟退火 (Simulated Annealing) 基准测试脚本
# 使用方式：从仓库根目录运行 python Resource_allocation_V2I/TD3/run_sa.py
# 核心逻辑：在每个 TimeSlot 内部运行 SA 循环，寻找当前时刻的最佳动作

import os
import sys
import time
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import math

# 确保可以导入同目录的模块
this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

import config as cfg
import Environment3
from Resource_allocation_V2I.SAC import dataStruct

# -------------------------- 参数设置 --------------------------
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
n_episode_test = 400  # 测试轮数

# SA 算法超参数 (可根据运行速度调整)
SA_ITER_PER_SLOT = 50  # 每个时隙内部迭代搜索次数 (越高越准但越慢)
T_INITIAL = 100.0  # 初始温度
T_FINAL = 0.1  # 终止温度
ALPHA = 0.90  # 降温系数
PERTURB_SCALE = 0.1  # 扰动幅度 (归一化范围的比例)

# 日志路径
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_base = f'../log/SA_numv{n_veh}_episode{n_episode_test}_{current_time}'
os.makedirs(log_base, exist_ok=True)


# -------------------------- SA 求解器 --------------------------
class SASolver:
    def __init__(self, n_veh, min_p, max_p, min_f, max_f):
        self.n_veh = n_veh
        self.min_p = min_p
        self.max_p = max_p
        self.min_f = min_f
        self.max_f = max_f

    def generate_initial_solution(self):
        """随机生成初始解 [n_veh, 3]"""
        action = np.zeros([self.n_veh, 3])
        for i in range(self.n_veh):
            action[i, 0] = np.random.uniform(self.min_p, self.max_p)
            action[i, 1] = np.random.uniform(self.min_f, self.max_f)
            action[i, 2] = np.random.uniform(0.0, 1.0)
        return action

    def get_neighbor(self, action):
        """生成邻域解（微小扰动）"""
        new_action = action.copy()
        for i in range(self.n_veh):
            # 随机选择一个维度进行扰动，或者扰动所有维度
            # 这里扰动所有维度

            # 1. Power
            delta_p = (self.max_p - self.min_p) * PERTURB_SCALE * np.random.uniform(-1, 1)
            new_action[i, 0] = np.clip(action[i, 0] + delta_p, self.min_p, self.max_p)

            # 2. Frequency
            delta_f = (self.max_f - self.min_f) * PERTURB_SCALE * np.random.uniform(-1, 1)
            new_action[i, 1] = np.clip(action[i, 1] + delta_f, self.min_f, self.max_f)

            # 3. Offload Rate
            delta_r = 1.0 * PERTURB_SCALE * np.random.uniform(-1, 1)
            new_action[i, 2] = np.clip(action[i, 2] + delta_r, 0.0, 1.0)

        return new_action

    def evaluate(self, env, action, time_slot_now):
        """
        评估动作的好坏 (计算 Reward)。
        【关键】：必须保护环境状态，不能真的 step 环境。
        """
        # 1. 备份关键环境状态：Buffer
        buffer_backup = copy.deepcopy(env.ReplayB_v)

        try:
            # 2. 执行计算逻辑 (抄自 train_td3/Environment3)
            comp_n_list_true, comp_n_list = env.true_calculate_num(action)
            comp_n_list_RSU = env.calculate_num_RSU()

            # 注意：update_buffer 会修改 buffer，但我们在 try-finally 中会恢复
            env.update_buffer(comp_n_list)

            offload_num = []
            for i in range(self.n_veh):
                offload_num_i = int(action[i, 2] * comp_n_list_RSU)
                offload_num.append(offload_num_i)

            # 获取当前信道 (不生成新的，只获取当前的)
            h_i_dB = env.overall_channel(time_slot_now)

            trans_energy_RSU = env.trans_energy_RSU(action, h_i_dB)

            # 计算 Reward
            # RSU_reward1 内部会修改 ReplayB_v (这是为什么要备份的原因)
            _, reward_tot, _, _, _ = env.RSU_reward1(
                action, comp_n_list_true, trans_energy_RSU, offload_num
            )

            # TD3 目标是最大化 reward = -1 * cost
            # 所以 SA 的能量值(Energy)定义为 cost，越小越好；或者定义为 reward，越大越好
            # 这里统一使用 reward (越大越好)
            current_reward = -1 * reward_tot
            return current_reward

        finally:
            # 3. 【恢复环境状态】
            env.ReplayB_v = buffer_backup


# -------------------------- 主逻辑 --------------------------
def save_results(name, index, E_total, reward, calculate, overload, eta1, load_rate_0, delay):
    """日志绘图逻辑，沿用现有结构"""
    log_dir = f'../log/{name}_numv{n_veh}_episode{n_episode_test}_{current_time}'
    os.makedirs(log_dir, exist_ok=True)
    data = {
        'Sum_E_total': E_total, 'Sum_reward': reward, 'Sum_calculate': calculate,
        'Sum_overload': overload, 'Sum_eta1': eta1, 'Sum_load_rate_0': load_rate_0, 'Sum_delay': delay
    }
    with open(f'{log_dir}/{name}_data_{index}.pkl', 'wb') as f:
        pickle.dump(data, f)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    metrics = {
        '能量消耗': E_total, '奖励': reward, '计算量': calculate,
        '过载量': overload, '资源浪费率': eta1, '卸载率': load_rate_0, '时延': delay
    }
    x = np.arange(len(E_total))
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        # SA 用绿色线条表示
        plt.plot(x, values, label='SA数据', alpha=0.6, color='green')
        window_size = 10
        if len(values) >= window_size:
            smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='same')
            plt.plot(x, smoothed, label=f'平滑曲线（窗口={window_size}）', color='darkgreen')
        plt.xlabel('测试轮次 (Episode)')
        plt.ylabel(metric_name)
        plt.title(f'{name}算法的{metric_name}趋势')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'{log_dir}/{name}_{metric_name}.png', bbox_inches='tight')
        plt.close()


def run_sa_once(index=0):
    # 初始化环境
    up_lanes = [i / 2.0 for i in [400 + 3.5 / 2, 400 + 3.5 + 3.5 / 2, 800 + 3.5 / 2, 800 + 3.5 + 3.5 / 2]]
    down_lanes = [i / 2.0 for i in [400 - 3.5 - 3.5 / 2, 400 - 3.5 / 2, 800 - 3.5 - 3.5 / 2, 800 - 3.5 / 2]]
    left_lanes = [i / 2.0 for i in [400 + 3.5 / 2, 400 + 3.5 + 3.5 / 2, 800 + 3.5 / 2, 800 + 3.5 + 3.5 / 2]]
    right_lanes = [i / 2.0 for i in [400 - 3.5 - 3.5 / 2, 400 - 3.5 / 2, 800 - 3.5 - 3.5 / 2, 800 - 3.5 / 2]]

    env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, 0, BS_width)
    env.new_random_game()

    # 初始化 SA 求解器
    sa = SASolver(n_veh, min_power, max_power, min_f, max_f)

    # 统计数据
    Sum_E_total_list = []
    Sum_reward_list = []
    Sum_calculate_list = []
    Sum_overload_list = []
    Sum_eta1_list = []
    Sum_load_rate_0_episode_list = []
    Sum_delay_list = []
    elapsed_list = []

    print(f"Start Simulated Annealing (SA) for {n_episode_test} episodes...")

    for i_episode in range(n_episode_test):
        start_time = time.time()
        env.new_random_game()

        # 这里的 state 仅用于遵循流程，SA 实际上不使用神经网络的状态输入
        state = env.get_state()

        Sum_E_total_per_episode = []
        Sum_reward_per_episode = []
        Sum_calculate_per_episode = []
        Sum_overload_per_episode = []
        Sum_delay_per_episode = []
        Sum_load_rate_0_episode = []
        eta1 = []

        time_slots = dataStruct.timeSlots(start=0, end=299, slot_length=1)

        # --- Time Slot Loop ---
        while not time_slots.is_end():
            current_t_slot = time_slots.now()

            # =========== SA 核心搜索过程 (每个时隙运行一次) ===========
            # 1. 初始解
            current_action = sa.generate_initial_solution()
            current_reward = sa.evaluate(env, current_action, current_t_slot)

            best_action = current_action.copy()
            best_reward = current_reward

            temperature = T_INITIAL

            # 2. 退火循环
            # 为了保证速度，这里使用固定次数迭代或者直到温度冷却
            # 这里结合了固定次数和温度冷却，保证每步搜索量有限
            for _ in range(SA_ITER_PER_SLOT):
                if temperature < T_FINAL:
                    break

                # 生成邻域
                neighbor_action = sa.get_neighbor(current_action)
                neighbor_reward = sa.evaluate(env, neighbor_action, current_t_slot)

                # Metropolis 准则
                # 我们要最大化 Reward，所以 delta = new - current
                delta = neighbor_reward - current_reward

                if delta > 0:
                    # 更好的解，直接接受
                    current_action = neighbor_action
                    current_reward = neighbor_reward
                    # 更新历史最优
                    if current_reward > best_reward:
                        best_reward = current_reward
                        best_action = current_action.copy()
                else:
                    # 较差的解，概率接受
                    # exp(delta / T)，因为 delta 是负数，所以结果是 (0, 1]
                    # 注意防止溢出
                    try:
                        prob = math.exp(delta / temperature)
                    except OverflowError:
                        prob = 0

                    if random.random() < prob:
                        current_action = neighbor_action
                        current_reward = neighbor_reward

                # 降温
                temperature *= ALPHA

            # 搜索结束，使用 best_action 执行真正的 Step
            final_action = best_action
            # ========================================================

            # 执行真正的环境交互 (Commit the action)
            comp_n_list_true, comp_n_list = env.true_calculate_num(final_action)
            comp_n_list_RSU = env.calculate_num_RSU()
            env.update_buffer(comp_n_list)  # 此时真正更新 Buffer

            offload_num = []
            for i in range(n_veh):
                offload_num_i = int(final_action[i, 2] * comp_n_list_RSU)
                offload_num.append(offload_num_i)

            h_i_dB = env.overall_channel(time_slots.now())
            trans_energy_RSU = env.trans_energy_RSU(final_action, h_i_dB)

            E_total, reward_tot, overload, load_rate_0, Delay_vel = env.RSU_reward1(
                final_action, comp_n_list_true, trans_energy_RSU, offload_num
            )  # 此时真正产生 Reward 和 Buffer 消耗

            # 记录数据
            eta1.append(overload / sum(comp_n_list) if sum(comp_n_list) != 0 else 0.0)
            if load_rate_0 == []:
                load_rate_0 = np.ones(n_veh)

            reward = -1 * reward_tot

            Sum_load_rate_0_episode.append(np.mean(load_rate_0))
            Sum_E_total_per_episode.append(np.sum(E_total))
            Sum_reward_per_episode.append(np.sum(reward))
            Sum_calculate_per_episode.append(np.round(np.sum(comp_n_list)))
            Sum_overload_per_episode.append(overload)
            Sum_delay_per_episode.append(sum(Delay_vel))

            # 更新状态和时间
            state_new = env.get_state()
            time_slots.add_time()

        # Episode 数据汇总
        elapsed = time.time() - start_time
        avg_reward = np.mean(Sum_reward_per_episode)
        elapsed_list.append(elapsed)
        print(f'------ SA/Episode {i_episode} | Time: {elapsed:.2f}s | Avg Reward: {avg_reward:.4f} ------')
        print('Sum_energy_per_episode:', round(np.average(Sum_E_total_per_episode), 6))
        print('Sum_reward_per_episode:', round(np.average(Sum_reward_per_episode), 6))
        print('Sum_calculate_per_episode:', round(np.average(Sum_calculate_per_episode)))
        print('Sum_overload_rate_per_episode:', round(np.average(eta1), 6))
        print('Sum_load_rate_0_episode:', round(np.average(Sum_load_rate_0_episode), 6))
        print('Sum_delay_per_episode:', round(np.average(Sum_delay_per_episode), 6))

        Sum_E_total_list.append(np.mean(Sum_E_total_per_episode))
        Sum_reward_list.append(avg_reward)
        Sum_calculate_list.append(np.round(np.mean(Sum_calculate_per_episode)))
        Sum_overload_list.append(np.round(np.mean(Sum_overload_per_episode)))
        Sum_eta1_list.append(np.mean(eta1))
        Sum_load_rate_0_episode_list.append(np.mean(Sum_load_rate_0_episode))
        Sum_delay_list.append(np.mean(Sum_delay_per_episode))

    save_results('SA', 0, Sum_E_total_list, Sum_reward_list, Sum_calculate_list,
                 Sum_overload_list, Sum_eta1_list, Sum_load_rate_0_episode_list, Sum_delay_list)
    print(f"SA benchmark finished. Results saved to: {log_base}")



if __name__ == "__main__":
    run_sa_once()