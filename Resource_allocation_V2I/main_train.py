import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from Resource_allocation_V2I import dataStruct

os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import Environment3
from RL_train5 import SAC_Trainer
from RL_train5 import ReplayBuffer

"""Time slot related."""
time_slot_start: int = 0
time_slot_end: int = 299
time_slot_number: int = 300
time_slot_length: int = 1

# -------------------------- 场景参数配置 --------------------------
draw_flag=False
# 基站（BS）宽度（单位：米，除以2是为了坐标计算）
BS_width = 1000/2
# 定义道路车道的坐标（上下左右四个方向，每个方向4条车道）
# 车道坐标计算基于道路宽度（3.5米），用于车辆初始位置生成
up_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]
left_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]
print(up_lanes)
print(down_lanes)
print(left_lanes)
print(right_lanes)

# 场景尺寸（1000x1000米）
width = 1000
height = 1000
# 通信与物理参数
B = 2e6# 带宽（Hz）
sig1=-144  # 噪声功率：dB
sig2=10**(sig1/10)# 噪声功率转换为线性值（W）
q_tao = 0.5# 信道相干时间（秒）

BS_position = [0, 0]# 基站坐标
max_power = 200# 车辆最大发射功率（单位未明确，可能为mW）
min_power = 5# 车辆最小发射功率
max_f = 4e8# 最大计算频率（Hz）
min_f = 5e7# 最小计算频率
m = 0.023  # 路径损耗系数（dB）


V2I_min = 3.16 # V2I通信的最小阈值（可能为信噪比门限）


# -------------------------- 训练参数配置 --------------------------
batch_size = 64# 每次训练的样本批次大小
memory_size = 1000000# 经验回放缓冲区的最大容量


n_step_per_episode = 100# 每轮训练的步数
# n_episode_test = 3000  # 总训练轮数（大循环次数）
n_episode_test = 5  #原论文为3000
n_interference_vehicle = 0# 干扰车辆数量（当前设置为0，无干扰）
n_veh =5 # 车辆数量


# ---------------------------状态和动作空间维度配置--------------------------------------------------------------------------------------- #
n_input = 2 * n_veh  # 状态输入维度（每个车辆2个特征，如位置坐标）
n_output = 3 * n_veh# 动作输出维度（每个车辆3个动作：功率、频率、卸载比例）

n_input_RSU = 2 * n_veh  # RSU（路侧单元）的状态输入维度
n_output_RSU = 2 * n_veh # RSU的动作输出维度


# -------------------------SAC算法参数-------------------------------------
replay_buffer_size = 1e6 # 经验回放缓冲区大小
replay_buffer_size_RSU = 1e6# RSU的经验回放缓冲区大小

hidden_dim = 512 # 神经网络隐藏层维度
action_range = 1.0# 动作范围（用于归一化）
AUTO_ENTROPY = True# 是否自动调整熵参数（探索-利用平衡）
DETERMINISTIC = False# 是否使用确定性策略（训练时设为False，测试时可设为True）

#--------------------- 其他强化学习参数（部分用于PPO，但当前主要用SAC）------------------------------------------
update_timestep = 100  # 每100步更新一次策略
action_std = 0.5  # 动作分布的标准差
K_epochs = 80  #  策略更新的迭代次数
eps_clip = 0.2  # PPO的剪辑参数
gamma = 0.99  # 折扣因子（未来奖励的衰减系数）
lr = 0.01  # 学习率
betas = (0.9, 0.999) # Adam优化器的动量参数

#------------------------------- 神经网络层维度（SAC中使用）--------------------------------
fc1_dims = 512
fc2_dims = 512
fc3_dims = 512
fc4_dims = 512
alpha = 0.0001# SAC中的熵温度参数学习率
beta = 0.001# 可能为其他损失项的系数
tau = 0.05# 目标网络软更新系数
pp = 0.005 # 可能为惩罚系数

def SAC_train(ii):
    """
        SAC算法训练函数：初始化环境、模型和缓冲区，执行多轮训练，返回训练指标。
        参数ii：可能用于多组实验的标识（当前未使用）
    """
    print("\nRestoring the SAC model...")
    # --------------model--------------
    model_path = 'model/SAC_model_episode1000_01'
    os.makedirs(model_path, exist_ok=True)  # exist_ok=True表示目录存在时不报错
    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(replay_buffer_size)
    # 初始化SAC训练器（传入缓冲区、状态/动作维度、隐藏层大小、动作范围）
    RL_SAC = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)

    # 存储训练过程中的关键指标
    Sum_E_total_list = []# 总能量消耗列表
    Sum_reward_list = [] # 总奖励列表
    Sum_calculate_list = []# 总计算任务量列表
    Sum_overload_list = []# 总过载量列表

    Sum_eta1_list = []# 过载率列表
    Sum_load_rate_0_episode_list = []# 负载率列表

    Sum_delay_list=[]# 总时延列表

    # 存储每辆车的位置轨迹
    Vehicle_positions_x = [[] for _ in range(n_veh)]
    Vehicle_positions_y = [[] for _ in range(n_veh)]

    # 主训练循环（共n_episode_test轮）
    for i_episode in range(n_episode_test):  #
        if i_episode ==50:
            print('A')
        print('------ SAC/Episode', i_episode, '------')
        # 初始化新一轮的环境（随机生成车辆初始位置等）
        # env.new_random_game()

        # 初始化新一轮的环境（读取CSV文件数据生成车辆初始位置等）
        env.new_random_game()

        if draw_flag:draw(env)

        # 记录当前轮所有车辆的初始位置
        # for i in range(n_veh):
        #     Vehicle_positions_x[i].append(env.vehicles[i].start_position[0])
        #     Vehicle_positions_y[i].append(env.vehicles[i].start_position[1])
        # 存储当前状态（初始状态）
        state_old_all = []
        # 存储当前轮的指标（每步的累加）
        state = env.get_state()
        state_old_all.append(state)

        Sum_E_total_per_episode = []#能量消耗列表
        Sum_reward_per_episode = []#奖励列表
        Sum_calculate_per_episode = []#计算任务量列表
        Sum_overload_per_episode = []#过载量列表

        Sum_load_rate_0_episode = []# 负载率列表

        Sum_delay_per_episode = []  # 时延列表

        eta1 = [] # 每步的过载率

        time_slots: dataStruct.timeSlots = dataStruct.timeSlots(
            start=time_slot_start,
            end=time_slot_end,
            slot_length=time_slot_length,
        )
        while(not time_slots.is_end()):
        # for i_step in range(time_slots.get_number()):
            # 更新车辆位置（模拟车辆移动）
            # env.renew_position()

            if draw_flag:draw(env)

            # 存储新状态和动作
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_veh, 3], dtype=np.float64)# 每个车辆3个动作
            # 根据当前状态生成动作（通过SAC的策略网络）
            # 状态展平为一维数组输入网络，deterministic=False表示随机探索
            action = RL_SAC.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            # 裁剪动作到[-0.999, 0.999]（避免极端值）
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)
            # 将网络输出的归一化动作映射到实际物理范围
            for i in range(n_veh):
                # 动作0：发射功率（映射到[min_power, max_power]）
                action_all_training[i, 0] = ((action[0 + i * 2] + 1) / 2) * (max_power-min_power)+min_power
                # 动作1：计算频率（映射到[min_f, max_f]）
                action_all_training[i, 1] = ((action[1 + i * 2] + 1) / 2) * (max_f-min_f)+min_f
                # 动作2：任务卸载比例（映射到[0, 1]）
                action_all_training[i, 2] = (action[2 + i * 2] + 1) / 2
            # 复制动作用于后续处理
            action_pf = action_all_training.copy()
            # 真实计算次数和理论计算次数（基于动作中的频率等参数）
            comp_n_list_true, comp_n_list = env.true_calculate_num(action_pf)
            # 计算RSU（路侧单元）可处理的任务量
            comp_n_list_RSU = env.calculate_num_RSU()
            # 更新缓冲区（用于记录任务队列状态）
            env.update_buffer(comp_n_list)
            # 计算每个车辆的卸载任务数（基于卸载比例和RSU处理能力）
            offload_num = []
            for i in range(n_veh):
                # 计算车辆卸载到RSU的任务数
                offload_num_i = int(action_pf[i, 2]*comp_n_list_RSU)#第 i 辆车的卸载比例（取值范围为 [0, 1]）
                offload_num.append(offload_num_i)
            # 计算每辆车信道增益（dB）
            h_i_dB= env.overall_channel(time_slots.now())
            # 计算每辆车到RSU的传输能量
            trans_energy_RSU = env.trans_energy_RSU(action_pf, h_i_dB)
            # 计算总能量消耗、奖励、过载量、卸载率、时延等
            #comp_n_list_true车辆在当前时间片内实际能处理的任务个数(单位：个)   offload_num每辆车计划卸载到 RSU（路侧单元）的任务数量(单位：个)
            E_total, reward_tot, overload, load_rate_0,Delay_vel  = env.RSU_reward1(action_pf, comp_n_list_true, trans_energy_RSU, offload_num)
            # 记录过载率（过载量/总任务量）
            eta1.append(overload/sum(comp_n_list))
            # 处理负载率为空的情况（默认设为1）
            if load_rate_0==[]:
                load_rate_0 = np.ones(n_veh)

            reward = -1 * reward_tot
            E_total = 1 * E_total

            # 记录当前步的指标
            Sum_load_rate_0_episode.append(np.mean(load_rate_0))
            Sum_E_total_per_episode.append(np.sum(E_total))
            Sum_reward_per_episode.append(np.sum(reward))
            Sum_calculate_per_episode.append(np.round(np.sum(comp_n_list)))
            Sum_overload_per_episode.append(overload)
            Sum_delay_per_episode.append(sum(Delay_vel))

            # 获取新状态（更新后的环境状态）
            state_new = env.get_state()
            state_new_all.append((state_new))

            # 将当前经验（状态、动作、奖励、新状态、是否结束）存入回放缓冲区  done设为0表示未结束（每步都继续）
            replay_buffer.push(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                           reward, np.asarray(state_new_all).flatten(), 0)
            # 当缓冲区数据量超过批次大小时，更新SAC模型
            if len(replay_buffer) > 256:   # 256为经验数量阈值
                for i in range(1):# 每次更新1次
                    # 调用SAC的更新函数，自动调整熵参数
                    _ = RL_SAC.update(batch_size,
                                      reward_scale=10.,
                                      auto_entropy=AUTO_ENTROPY,
                                      target_entropy=-1.*n_output) # 目标熵（与动作维度相关）
            # 更新状态（进入下一步）
            state_old_all = state_new_all

            time_slots.add_time()
        # 记录当前轮的平均指标
        Sum_E_total_list.append((np.mean(Sum_E_total_per_episode)))
        Sum_reward_list.append((np.mean(Sum_reward_per_episode)))
        Sum_calculate_list.append(np.round(np.mean(Sum_calculate_per_episode)))
        Sum_overload_list.append(np.round(np.mean(Sum_overload_per_episode)))

        Sum_eta1_list.append((np.mean(eta1)))
        Sum_load_rate_0_episode_list.append((np.mean(Sum_load_rate_0_episode)))

        Sum_delay_list.append(np.mean(Sum_delay_per_episode))



        print('Sum_energy_per_episode:', round(np.average(Sum_E_total_per_episode), 6))
        print('Sum_reward_per_episode:', round(np.average(Sum_reward_per_episode), 6))
        print('Sum_calculate_per_episode:', round(np.average(Sum_calculate_per_episode)))
        print('Sum_overload_rate_per_episode:', round(np.average(eta1), 6))
        print('Sum_load_rate_0_episode:', round(np.average(Sum_load_rate_0_episode),6))
        print('Sum_delay_per_episode:', round(np.average(Sum_delay_per_episode), 6))



    # 训练结束后保存模型参数
    RL_SAC.save_model(model_path)

    return Sum_E_total_list, Sum_reward_list, Sum_calculate_list, Sum_overload_list, Sum_eta1_list, Sum_load_rate_0_episode_list,Sum_delay_list


def save_results(name, index, E_total, reward, calculate, overload, eta1, load_rate_0,delay):
    """
    保存训练数据并绘制图表
    :param name: 算法名称（如'SAC'）
    :param index: 实验索引
    :param E_total: 每轮能量列表
    :param reward: 每轮奖励列表
    :param calculate: 每轮计算量列表
    :param overload: 每轮过载量列表
    :param eta1: 每轮过载率列表
    :param load_rate_0: 每轮卸载率率列表
    :param delay: 每轮时延列表
    """
    # 创建保存目录（如log/SAC_1/）
    log_dir = f'log/{name}_{index}'
    os.makedirs(log_dir, exist_ok=True)

    # 设置中文字体 使用 SimHei 黑体（系统中有）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # SimHei 优先，英文字体备用
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 1. 保存原始数据（pickle格式）
    data = {
        'Sum_E_total': E_total,
        'Sum_reward': reward,
        'Sum_calculate': calculate,
        'Sum_overload': overload,
        'Sum_eta1': eta1,
        'Sum_load_rate_0': load_rate_0,
        'Sum_delay':delay
    }
    with open(f'{log_dir}/{name}_data_{index}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"数据已保存至 {log_dir}/{name}_data_{index}.pkl")

    # 2. 绘制并保存图表
    metrics = {
        '能量消耗': E_total,
        '奖励': reward,
        '计算量': calculate,
        '过载量': overload,
        '过载率': eta1,
        '卸载率': load_rate_0,
        '时延':delay
    }
    x = np.arange(len(E_total))  # 横轴为训练轮次（episode）

    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        # 绘制原始曲线
        plt.plot(x, values, label='原始数据', alpha=0.6)
        # 绘制平滑曲线（移动平均，窗口大小为10）
        window_size = 10
        if len(values) >= window_size:
            smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='same')
            plt.plot(x, smoothed, label=f'平滑曲线（窗口={window_size}）', color='red')
        plt.xlabel('训练轮次 (Episode)')
        plt.ylabel(metric_name)
        plt.title(f'{name}算法的{metric_name}趋势')
        plt.legend()
        plt.grid(alpha=0.3)
        # 保存图表（PDF格式，矢量图适合论文）
        plt.savefig(f'{log_dir}/{name}_{metric_name}_{index}.png', bbox_inches='tight')
        plt.close()  # 关闭图表，避免内存占用
    print(f"图表已保存至 {log_dir} 目录")


def draw(env):
    lane_width = 3.5  # 车道宽度（代码中隐含的3.5米）

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # SimHei 优先，英文字体备用
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    """绘制环境模型和车辆位置"""
    plt.figure(figsize=(10, 10))
    plt.xlim(0, width)
    plt.ylim(0, height)

    # 绘制上行车道（水平车道，y为中心）
    # for y in up_lanes:
    #     plt.fill_between(
    #         [0, width],
    #         y - lane_width / 2,
    #         y + lane_width / 2,
    #         color='lightgreen',
    #         alpha=0.5,
    #         label='上行车道' if y == up_lanes[0] else ""
    #     )
    #     plt.text(width + 10, y, f'y={y}', verticalalignment='center')

    plt.fill_betweenx(
        [0, width],
        up_lanes[0],
        up_lanes[1],
        color='lightgreen',
        alpha=0.5,
        label='上行车道1'
    )
    plt.text(up_lanes[0], height + 10, f'x={up_lanes[0]}', horizontalalignment='center')
    plt.text(up_lanes[1], height + 10, f'x={up_lanes[1]}', horizontalalignment='center')

    plt.fill_betweenx(
        [0, width],
        up_lanes[2],
        up_lanes[3],
        color='lightgreen',
        alpha=0.5,
        label='上行车道2'
    )
    plt.text(up_lanes[2], height + 10, f'x={up_lanes[2]}', horizontalalignment='center')
    plt.text(up_lanes[3], height + 10, f'x={up_lanes[3]}', horizontalalignment='center')

    # 绘制下行车道
    # for y in down_lanes:
    #     plt.fill_between(
    #         [0, width],
    #         y - lane_width / 2,
    #         y + lane_width / 2,
    #         color='lightblue',
    #         alpha=0.5,
    #         label='下行车道' if y == down_lanes[0] else ""
    #     )
    #     plt.text(width + 10, y, f'y={y}', verticalalignment='center')

    plt.fill_betweenx(
        [0, height],
        down_lanes[0],
        down_lanes[1],
        color='lightblue',
        alpha=0.5,
        label='下行车道1'
    )
    plt.text(down_lanes[0], height + 10, f'x={down_lanes[0]}', horizontalalignment='center')
    plt.text(down_lanes[1], height + 10, f'x={down_lanes[1]}', horizontalalignment='center')

    plt.fill_betweenx(
        [0, height],
        down_lanes[2],
        down_lanes[3],
        color='lightblue',
        alpha=0.5,
        label='下行车道2'
    )
    plt.text(down_lanes[2], height + 10, f'x={down_lanes[3]}', horizontalalignment='center')
    plt.text(down_lanes[2], height + 10, f'x={down_lanes[3]}', horizontalalignment='center')


    # 绘制左车道（垂直车道，x为中心）
    # for x in left_lanes:
    #     plt.fill_betweenx(
    #         [0, height],
    #         x - lane_width / 2,
    #         x + lane_width / 2,
    #         color='lightcoral',
    #         alpha=0.5,
    #         label='左车道' if x == left_lanes[0] else ""
    #     )
    #     plt.text(x, height + 10, f'x={x}', horizontalalignment='center')

    plt.fill_between(
        [0, width],
        left_lanes[0],
        left_lanes[1],
        color='lightcoral',
        alpha=0.5,
        label='左车道1'
    )
    plt.text(width + 10, left_lanes[0], f'y={left_lanes[0]}', verticalalignment='center')
    plt.text(width + 10, left_lanes[1], f'y={left_lanes[1]}', verticalalignment='center')

    plt.fill_between(
        [0, width],
        left_lanes[2],
        left_lanes[3],
        color='lightcoral',
        alpha=0.5,
        label='左车道2'
    )
    plt.text(width + 10, left_lanes[2], f'y={left_lanes[2]}', verticalalignment='center')
    plt.text(width + 10, left_lanes[3], f'y={left_lanes[3]}', verticalalignment='center')

    # 绘制右车道
    # for x in right_lanes:
    #     plt.fill_betweenx(
    #         [0, height],
    #         x - lane_width / 2,
    #         x + lane_width / 2,
    #         color='lightsalmon',
    #         alpha=0.5,
    #         label='右车道' if x == right_lanes[0] else ""
    #     )
    #     plt.text(x, height + 10, f'x={x}', horizontalalignment='center')

    plt.fill_between(
        [0, width],
        right_lanes[0],
        right_lanes[1],
        color='lightsalmon',
        alpha=0.5,
        label='右车道1'
    )
    plt.text(width + 10, right_lanes[0], f'y={right_lanes[0]}', verticalalignment='center')
    plt.text(width + 10, right_lanes[1], f'y={right_lanes[1]}', verticalalignment='center')

    plt.fill_between(
        [0, width],
        right_lanes[2],
        right_lanes[3],
        color='lightsalmon',
        alpha=0.5,
        label='右车道2'
    )
    plt.text(width + 10, right_lanes[2], f'y={right_lanes[2]}', verticalalignment='center')
    plt.text(width + 10, right_lanes[3], f'y={right_lanes[3]}', verticalalignment='center')

    # 绘制车辆
    dir_marker = {'u': '↑', 'd': '↓', 'l': '←', 'r': '→'}
    for vehicle in env.vehicles:
        p=vehicle.start_position
        d=vehicle.direction
        plt.scatter(p[0], p[1], s=50, c='black', zorder=3)
        plt.text(p[0] + 5, p[1] + 5, dir_marker[d], fontsize=12)

    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('车道环境模型与随机车辆位置')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


if __name__ == "__main__":


    for i in range(1):
        name = 'SAC'
        # 初始化环境（传入车道参数、场景尺寸、车辆数量等）
        env = Environment3.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_interference_vehicle, BS_width)
        # 初始化环境状态（随机生成车辆位置等）
        env.new_random_game()


        # E_total_list, reward_list, calculate_list, overload_list, buffer_list, eta1_list, load_rate_0_episode_list = SAC_train(i)
        #
        # save_results(name, i, E_total_list, reward_list, calculate_list, overload_list, buffer_list, eta1_list, load_rate_0_episode_list)

        E_total_list, reward_list, calculate_list, overload_list, eta1_list, load_rate_0_episode_list,Sum_delay_list = SAC_train(i)
        # 调用保存函数
        save_results(name, i, E_total_list, reward_list, calculate_list, overload_list, eta1_list, load_rate_0_episode_list,Sum_delay_list)