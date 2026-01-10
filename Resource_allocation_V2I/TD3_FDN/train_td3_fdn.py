# TD3_FDN 训练入口
import os
import sys
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt

this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

from td3_agent_fdn import TD3_FDN_Agent, ReplayBuffer
import config as cfg

import Environment3
from Resource_allocation_V2I.SAC import dataStruct

# 场景参数
n_veh = cfg.n_veh
n_input = cfg.n_input
n_output = cfg.n_output

BS_width = cfg.BS_width if hasattr(cfg, 'BS_width') else 1000/2
width = cfg.width if hasattr(cfg, 'width') else 1000
height = cfg.height if hasattr(cfg, 'height') else 1000
max_power = cfg.max_power
min_power = cfg.min_power
max_f = cfg.max_f
min_f = cfg.min_f

batch_size = cfg.batch_size
memory_size = cfg.memory_size
n_step_per_episode = 100
n_episode_test = 400
DETERMINISTIC = False  # 评估时设 True

hidden_dim = cfg.hidden_dim
action_range = cfg.action_range

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = './model/TD3_FDN_episode' + str(n_episode_test) + "_" + current_time
os.makedirs(model_path, exist_ok=True)

log_base = f'../log/TD3_FDN_episode{n_episode_test}_{current_time}'
os.makedirs(log_base, exist_ok=True)

replay_buffer = ReplayBuffer(memory_size)
agent = TD3_FDN_Agent(
    state_dim=n_input,
    action_dim=n_output,
    hidden_dim=hidden_dim,
    action_range=action_range,
    replay_buffer=replay_buffer,
    actor_lr=cfg.actor_lr,
    critic_lr=cfg.critic_lr,
    gamma=cfg.gamma,
    tau=cfg.tau,
    policy_noise=cfg.policy_noise,
    noise_clip=cfg.noise_clip,
    policy_delay=cfg.policy_delay,
    denoising_steps=cfg.denoising_steps,
    t_dim=cfg.t_dim,
    beta_schedule=cfg.beta_schedule,
    fdn_noise_scale=cfg.fdn_noise_scale,
)

def save_results(name, index, E_total, reward, calculate, overload, eta1, load_rate_0, delay):
    log_dir = f'../log/{name}_episode{n_episode_test}_{current_time}'
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
    with open(f'{log_dir}/{name}_data_{index}.pkl', 'wb') as f:
        pickle.dump(data, f)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    metrics = {
        '能量消耗': E_total,
        '奖励': reward,
        '计算量': calculate,
        '过载量': overload,
        '资源浪费率': eta1,
        '卸载率': load_rate_0,
        '时延': delay
    }
    x = np.arange(len(E_total))
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(x, values, label='原始数据', alpha=0.6)
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

def train_td3_fdn_once(index=0):
    up_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
    down_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]
    left_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
    right_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]

    env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, 0, BS_width)
    env.new_random_game()

    Sum_E_total_list, Sum_reward_list, Sum_calculate_list = [], [], []
    Sum_overload_list, Sum_eta1_list, Sum_load_rate_0_episode_list, Sum_delay_list = [], [], [], []

    for i_episode in range(n_episode_test):
        print('------ TD3_FDN/Episode', i_episode, '------')
        env.new_random_game()

        state_old_all = []
        state = env.get_state()
        state_old_all.append(state)

        Sum_E_total_per_episode = []
        Sum_reward_per_episode = []
        Sum_calculate_per_episode = []
        Sum_overload_per_episode = []
        Sum_delay_per_episode = []
        Sum_load_rate_0_episode = []
        eta1 = []

        time_slots = dataStruct.timeSlots(start=0, end=299, slot_length=1)
        last_actions = []  # 存历史动作（上一时隙）

        while not time_slots.is_end():
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, 3], dtype=np.float64)

            # 使用扩散策略生成动作；探索通过 fdn_noise_scale 控制
            action = agent.get_action(np.asarray(state_old_all).flatten(),
                                      deterministic=DETERMINISTIC,
                                      history_action=last_actions[time_slots.now()-1] if time_slots.now()!=0 else None)
            action = np.clip(action, -0.999, 0.999)

            action_all.append(action)
            last_actions.append(action)

            # 映射到物理范围（每车3个动作）
            for i in range(n_veh):
                idx = i * 3
                action_all_training[i, 0] = ((action[idx + 0] + 1) / 2) * (max_power - min_power) + min_power
                action_all_training[i, 1] = ((action[idx + 1] + 1) / 2) * (max_f - min_f) + min_f
                action_all_training[i, 2] = (action[idx + 2] + 1) / 2

            comp_n_list_true, comp_n_list = env.true_calculate_num(action_all_training)
            comp_n_list_RSU = env.calculate_num_RSU()
            env.update_buffer(comp_n_list)
            offload_num = []
            for i in range(n_veh):
                offload_num_i = int(action_all_training[i, 2] * comp_n_list_RSU)
                offload_num.append(offload_num_i)
            h_i_dB = env.overall_channel(time_slots.now())
            trans_energy_RSU = env.trans_energy_RSU(action_all_training, h_i_dB)
            E_total, reward_tot, overload, load_rate_0, Delay_vel = env.RSU_reward1(action_all_training, comp_n_list_true, trans_energy_RSU, offload_num)

            eta1.append(overload / sum(comp_n_list) if sum(comp_n_list) != 0 else 0.0)
            if load_rate_0 == []:
                load_rate_0 = np.ones(n_veh)

            reward = -1 * reward_tot
            E_total = 1 * E_total

            Sum_load_rate_0_episode.append(np.mean(load_rate_0))
            Sum_E_total_per_episode.append(np.sum(E_total))
            Sum_reward_per_episode.append(np.sum(reward))
            Sum_calculate_per_episode.append(np.round(np.sum(comp_n_list)))
            Sum_overload_per_episode.append(overload)
            Sum_delay_per_episode.append(sum(Delay_vel))

            state_new = env.get_state()
            state_new_all.append(state_new)

            # push to replay buffer
            replay_buffer.push(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                               reward, np.asarray(state_new_all).flatten(), 0)

            # 更新 Agent
            if len(replay_buffer) > 256:
                critic_loss, actor_loss = agent.update(batch_size=batch_size)

            state_old_all = state_new_all
            time_slots.add_time()

        Sum_E_total_list.append(np.mean(Sum_E_total_per_episode))
        Sum_reward_list.append(np.mean(Sum_reward_per_episode))
        Sum_calculate_list.append(np.round(np.mean(Sum_calculate_per_episode)))
        Sum_overload_list.append(np.round(np.mean(Sum_overload_per_episode)))
        Sum_eta1_list.append(np.mean(eta1))
        Sum_load_rate_0_episode_list.append(np.mean(Sum_load_rate_0_episode))
        Sum_delay_list.append(np.mean(Sum_delay_per_episode))

        print('Sum_energy_per_episode:', round(np.average(Sum_E_total_per_episode), 6))
        print('Sum_reward_per_episode:', round(np.average(Sum_reward_per_episode), 6))
        print('Sum_calculate_per_episode:', round(np.average(Sum_calculate_per_episode)))
        print('Sum_overload_rate_per_episode:', round(np.average(eta1), 6))
        print('Sum_load_rate_0_episode:', round(np.average(Sum_load_rate_0_episode), 6))
        print('Sum_delay_per_episode:', round(np.average(Sum_delay_per_episode), 6))

    agent.save_model(model_path)
    save_results('TD3_FDN', 0, Sum_E_total_list, Sum_reward_list, Sum_calculate_list, Sum_overload_list, Sum_eta1_list, Sum_load_rate_0_episode_list, Sum_delay_list)
    return Sum_E_total_list, Sum_reward_list, Sum_calculate_list, Sum_overload_list, Sum_eta1_list, Sum_load_rate_0_episode_list, Sum_delay_list

if __name__ == "__main__":
    train_td3_fdn_once()
    print("TD3_FDN training finished. Models saved to:", model_path)