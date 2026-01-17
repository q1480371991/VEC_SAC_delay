import os
import sys
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt

this_dir = os.path.dirname(os.path.abspath(__file__))
if this_dir not in sys.path:
    sys.path.insert(0, this_dir)

import config as cfg
from matd3_agent import MATD3Agent
from multi_replay_buffer import MultiReplayBuffer

# 项目环境
import Environment3
from Resource_allocation_V2I.SAC import dataStruct

n_veh = cfg.n_veh
n_input = cfg.n_input
n_output = cfg.n_output

BS_width = cfg.BS_width
width = cfg.width
height = cfg.height
max_power = cfg.max_power
min_power = cfg.min_power
max_f = cfg.max_f
min_f = cfg.min_f

batch_size = 64
memory_size = cfg.memory_size
n_step_per_episode = 100
n_episode_test = 400
DETERMINISTIC = False

hidden_dim = cfg.hidden_dim
action_range = cfg.action_range

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'./model/MATD3_numv{n_veh}_episode{n_episode_test}_{current_time}'
os.makedirs(model_path, exist_ok=True)

log_base = f'../log/MATD3_numv{n_veh}_episode{n_episode_test}_{current_time}'
os.makedirs(log_base, exist_ok=True)

replay_buffer = MultiReplayBuffer(memory_size)
agent = MATD3Agent(
    n_veh=n_veh,
    obs_dim_per_agent=2,
    act_dim_per_agent=3,
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
    actor_shared=cfg.ACTOR_SHARED,
    critic_vector=cfg.CRITIC_VECTOR,
    use_gpu=cfg.USE_GPU,
    device_idx=cfg.DEVICE_IDX,
    rnn_type=cfg.RNN_TYPE,
    rnn_hidden_size=cfg.RNN_HIDDEN_SIZE,
)

def map_actions_to_physical(action_all_norm):
    action_all_training = np.zeros([n_veh, 3], dtype=np.float64)
    for i in range(n_veh):
        idx = i * 3
        action_all_training[i, 0] = ((action_all_norm[idx + 0] + 1) / 2) * (max_power - min_power) + min_power
        action_all_training[i, 1] = ((action_all_norm[idx + 1] + 1) / 2) * (max_f - min_f) + min_f
        action_all_training[i, 2] = (action_all_norm[idx + 2] + 1) / 2
    return action_all_training

def build_reward(shared_global_reward, E_total, Delay_vel, ReplayB_v, overload):
    mode = cfg.REWARD_MODE
    w = cfg.REWARD_WEIGHTS
    if mode == 'shared_global':
        r_shared = -float(shared_global_reward)
        return np.array([r_shared], dtype=np.float32)
    elif mode == 'shared_from_per_agent':
        energy_term = w['energy'] * float(np.mean(E_total))
        delay_term = w['delay'] * float(np.mean(Delay_vel))
        global_term = w['overload'] * float(overload) + w['buffer'] * float(np.sum(ReplayB_v))
        r_shared = -(energy_term + delay_term + global_term)
        return np.array([r_shared], dtype=np.float32)
    elif mode == 'per_agent':
        energy_vec = w['energy'] * np.asarray(E_total, dtype=np.float32)
        delay_vec = w['delay'] * np.asarray(Delay_vel, dtype=np.float32)
        global_share = (w['overload'] * float(overload) + w['buffer'] * float(np.sum(ReplayB_v))) / max(1, n_veh)
        r_agents = -(energy_vec + delay_vec + global_share)
        return r_agents.astype(np.float32)
    else:
        raise ValueError(f"Unknown REWARD_MODE: {mode}")

def save_results(name, index, E_total, reward, calculate, overload, eta1, load_rate_0, delay,
                 per_agent_energy_series=None, per_agent_delay_series=None):
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
        'PerAgent_energy_mean_per_episode': per_agent_energy_series,
        'PerAgent_delay_mean_per_episode': per_agent_delay_series,
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

    if per_agent_energy_series is not None and per_agent_delay_series is not None:
        episodes = np.arange(len(per_agent_energy_series[0]))
        plt.figure(figsize=(12, 7))
        for i in range(n_veh):
            plt.plot(episodes, per_agent_energy_series[i], label=f'Agent {i} 能量')
        plt.xlabel('训练轮次 (Episode)')
        plt.ylabel('逐智能体能量均值')
        plt.title(f'{name}逐智能体能量均值（每 Episode）')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'{log_dir}/{name}_per_agent_energy.png', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 7))
        for i in range(n_veh):
            plt.plot(episodes, per_agent_delay_series[i], label=f'Agent {i} 时延')
        plt.xlabel('训练轮次 (Episode)')
        plt.ylabel('逐智能体时延均值')
        plt.title(f'{name}逐智能体时延均值（每 Episode）')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f'{log_dir}/{name}_per_agent_delay.png', bbox_inches='tight')
        plt.close()

def train_matd3_once(index=0):
    up_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
    down_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]
    left_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
    right_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]

    env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, 0, BS_width)
    env.new_random_game()

    Sum_E_total_list = []
    Sum_reward_list = []
    Sum_calculate_list = []
    Sum_overload_list = []
    Sum_eta1_list = []
    Sum_load_rate_0_episode_list = []
    Sum_delay_list = []

    per_agent_energy_series = [ [] for _ in range(n_veh) ]
    per_agent_delay_series = [ [] for _ in range(n_veh) ]

    for i_episode in range(n_episode_test):
        print('------ MATD3 (RNN:{} / Shared Actor:{}) / Episode {} ------'.format(cfg.RNN_TYPE, cfg.ACTOR_SHARED, i_episode))
        env.new_random_game()
        agent.reset_hidden(batch_size=1)  # 重置在线隐状态

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

        per_agent_energy_acc = np.zeros(n_veh, dtype=np.float64)
        per_agent_delay_acc = np.zeros(n_veh, dtype=np.float64)
        per_agent_steps = 0

        time_slots = dataStruct.timeSlots(start=0, end=299, slot_length=1)
        while not time_slots.is_end():
            state_new_all = []
            action_all = []

            action_norm = agent.get_actions(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC, explore_noise=cfg.exploration_noise)
            action_norm = np.clip(action_norm, -0.999, 0.999)
            action_all.append(action_norm)

            action_all_training = map_actions_to_physical(action_norm)

            comp_n_list_true, comp_n_list = env.true_calculate_num(action_all_training)
            comp_n_list_RSU = env.calculate_num_RSU()
            env.update_buffer(comp_n_list)
            offload_num = [int(action_all_training[i, 2] * comp_n_list_RSU) for i in range(n_veh)]
            h_i_dB = env.overall_channel(time_slots.now())
            trans_energy_RSU = env.trans_energy_RSU(action_all_training, h_i_dB)
            E_total, reward_tot, overload, load_rate_0, Delay_vel = env.RSU_reward1(action_all_training, comp_n_list_true, trans_energy_RSU, offload_num)

            eta1.append(overload / sum(comp_n_list) if sum(comp_n_list) != 0 else 0.0)
            if load_rate_0 == []:
                load_rate_0 = np.ones(n_veh)

            reward = build_reward(shared_global_reward=reward_tot, E_total=E_total, Delay_vel=Delay_vel, ReplayB_v=env.ReplayB_v, overload=overload)

            Sum_load_rate_0_episode.append(np.mean(load_rate_0))
            Sum_E_total_per_episode.append(np.sum(E_total))
            Sum_reward_per_episode.append(np.sum(reward) if reward.shape[0] > 1 else float(reward[0]))
            Sum_calculate_per_episode.append(np.round(np.sum(comp_n_list)))
            Sum_overload_per_episode.append(overload)
            Sum_delay_per_episode.append(sum(Delay_vel))

            per_agent_energy_acc += np.asarray(E_total, dtype=np.float64)
            per_agent_delay_acc += np.asarray(Delay_vel, dtype=np.float64)
            per_agent_steps += 1

            state_new = env.get_state()
            state_new_all.append(state_new)

            replay_buffer.push(np.asarray(state_old_all).flatten(),
                               np.asarray(action_all).flatten(),
                               reward,
                               np.asarray(state_new_all).flatten(),
                               0)

            if len(replay_buffer) > 256:
                critic_loss, actor_loss = agent.update(batch_size=cfg.batch_size)

            state_old_all = state_new_all
            time_slots.add_time()

        Sum_E_total_list.append(np.mean(Sum_E_total_per_episode))
        Sum_reward_list.append(np.mean(Sum_reward_per_episode))
        Sum_calculate_list.append(np.round(np.mean(Sum_calculate_per_episode)))
        Sum_overload_list.append(np.round(np.mean(Sum_overload_per_episode)))
        Sum_eta1_list.append(np.mean(eta1))
        Sum_load_rate_0_episode_list.append(np.mean(Sum_load_rate_0_episode))
        Sum_delay_list.append(np.mean(Sum_delay_per_episode))

        if per_agent_steps > 0:
            per_agent_energy_mean = (per_agent_energy_acc / per_agent_steps).tolist()
            per_agent_delay_mean = (per_agent_delay_acc / per_agent_steps).tolist()
        else:
            per_agent_energy_mean = [0.0] * n_veh
            per_agent_delay_mean = [0.0] * n_veh

        for i in range(n_veh):
            per_agent_energy_series[i].append(per_agent_energy_mean[i])
            per_agent_delay_series[i].append(per_agent_delay_mean[i])

        print('Sum_energy_per_episode:', round(np.average(Sum_E_total_per_episode), 6))
        print('Sum_reward_per_episode:', round(np.average(Sum_reward_per_episode), 6))
        print('Sum_calculate_per_episode:', round(np.average(Sum_calculate_per_episode)))
        print('Sum_overload_rate_per_episode:', round(np.average(eta1), 6))
        print('Sum_load_rate_0_episode:', round(np.average(Sum_load_rate_0_episode), 6))
        print('Sum_delay_per_episode:', round(np.average(Sum_delay_per_episode), 6))

    agent.save_model(model_path)
    save_results('MATD3_RNN', 0,
                 Sum_E_total_list, Sum_reward_list, Sum_calculate_list, Sum_overload_list,
                 Sum_eta1_list, Sum_load_rate_0_episode_list, Sum_delay_list,
                 per_agent_energy_series=per_agent_energy_series,
                 per_agent_delay_series=per_agent_delay_series)

    return (Sum_E_total_list, Sum_reward_list, Sum_calculate_list, Sum_overload_list,
            Sum_eta1_list, Sum_load_rate_0_episode_list, Sum_delay_list)

if __name__ == "__main__":
    train_matd3_once()
    print("MATD3 training finished. Models saved to:", model_path)