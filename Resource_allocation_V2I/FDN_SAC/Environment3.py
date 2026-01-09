from __future__ import division

import dataclasses
from typing import List, Optional

import numpy as np
import random
import math
import dataStruct

from beta_allocation import BetaAllocation

"""Time slot related."""
time_slot_start: int = 0
time_slot_end: int = 299
time_slot_number: int = 300
time_slot_length: int = 1

"""Vehicle related."""
vehicle_number: Optional[int] = 5
trajectories_file_name: str = '../CSV/trajectories_20161116_2300_2305'
task_request_rate: float = 1
vehicle_seeds: List[int] = dataclasses.field(default_factory=list)
vehicle_seeds = [i for i in range(5)]

"""Task related."""
task_number: int = 100
task_minimum_data_size: float = 0.01 * 1024 * 1024 * 8 # 1 MB
task_maximum_data_size: float = 5 * 1024 * 1024 * 8 # 5 MB
task_minimum_computation_cycles: float = 500
task_maximum_computation_cycles: float = 500 # CPU cycles for processing 1-bit of data
task_minimum_delay_thresholds: float = 5 # seconds
task_maximum_delay_thresholds: float = 10 # seconds
task_seed: int = 0

""""Edge related."""
edge_number: int = 1
# edge_number: int = 9
edge_power: float = 1000.0 # mW
edge_bandwidth: float = 20.0  # MHz
edge_minimum_computing_cycles: float = 3.0 * 1e9 # 3 GHz
edge_maximum_computing_cycles: float = 10.0 * 1e9 # 10 GHz
communication_range: float = 500.0  # meters
map_length: float = 3000.0  # meters
map_width: float = 3000.0  # meters
edge_seed: int = 0


n_elements_total = 4# 总元素数量
n_elements = 4# 元素数量
t_trans_max =5# 最大传输时间
# RSU_position = [250, 250] # RSU（路侧单元）的位置坐标

RSU_position = [500, 500] # 根据CSV数据的edge坐标


class V2Ichannels:
    """V2I（车到基础设施）信道模型类，用于计算路径损耗和阴影衰落"""

    def __init__(self):
        self.h_bs = 25# 基站（RSU）天线高度（单位：米）
        self.h_ms = 1.5# 移动站（车辆）天线高度（单位：米）
        self.fc = 2# 载波频率（单位：GHz）
        self.Decorrelation_distance = 10# 去相关距离（单位：米），用于阴影衰落计算
        self.shadow_std = 8# 阴影衰落标准差（单位：dB）

    def get_path_loss(self, position_A):
        """
                计算从车辆到RSU的路径损耗
                :param position_A: 车辆的位置坐标 [x, y]
                :return: 路径损耗值（单位：dB）
        """
        RSU_location= dataStruct.location(x=RSU_position[0], y=RSU_position[1])
        distance=position_A.get_distance(RSU_location)

        # 计算车辆与RSU在x和y方向上的距离差
        # d1 = abs(position_A.get_x - RSU_position[0])
        # d2 = abs(position_A.get_y - RSU_position[1])
        # # 计算直线距离
        # distance = math.hypot(d1, d2)
        # 路径损耗模型公式（基于3GPP标准模型）
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)

    def get_shadowing(self, delta_distance, shadowing):
        """
                计算阴影衰落，考虑车辆间距离对阴影相关性的影响
                :param delta_distance: 车辆间的距离变化
                :param shadowing: 阴影衰落初始值
                :return: 更新后的阴影衰落值
        """
        nVeh = len(shadowing)# 车辆数量
        # 生成相关矩阵R，用于描述阴影衰落的相关性
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        # 阴影衰落更新公式，考虑距离衰减和随机扰动
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:
    """Vehicle simulator: include all the information for a Vehicle"""
    def __init__(self, start_position, start_direction, velocity):
        """车辆模拟器类，包含车辆的所有信息"""
        self.start_position = start_position# 初始位置 [x, y]
        self.direction = start_direction # 行驶方向（'u'上, 'd'下, 'l'左, 'r'右）
        self.velocity = velocity # 行驶速度
        self.neighbors = [] # 邻居车辆列表
        self.destinations = []# 目的地列表

class Environ:
    """环境类，用于模拟车辆行驶环境和通信场景"""
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_interference_vehicle, BS_width):
        self.time_slots: Optional[dataStruct.timeSlots] = None
        # 车道定义
        self.down_lanes = down_lane # 下行车道
        self.up_lanes = up_lane# 上行车道
        self.left_lanes = left_lane# 左行车道
        self.right_lanes = right_lane# 右行车道
        self.BS_width = BS_width# 基站覆盖宽度
        self.width = width# 环境宽度
        self.height = height# 环境高度
        self.interference_vehicle_num = n_interference_vehicle# 干扰车辆数量

        # 通信参数
        self.h_bs = 25# 基站天线高度（米）
        self.h_ms = 1.5# 移动站天线高度（米）
        self.fc = 2# 载波频率（GHz）
        self.T_n = 1 # 时间间隔（秒）
        self.sig2_dB = -114# 噪声功率（dBm）
        self.sig2 = 10 ** (self.sig2_dB / 10)   # 将噪声功率从dBm转换为瓦特（w）
        self.bsAntGain = 8# 基站天线增益（dB）
        self.bsNoiseFigure = 5# 基站噪声系数（dB）
        self.vehAntGain = 3 # 车辆天线增益（dB）
        self.vehNoiseFigure = 11# 车辆噪声系数（dB）

        self.n_veh = n_veh # 车辆数量
        self.V2Ichannels = V2Ichannels()# 实例化V2I信道模型
        self.vehicle_list=[]
        self.vehicles = []# 车辆列表

        # 存储各类信息的列表
        self.demand = []# 车辆的通信需求
        self.V2I_Shadowing = []# V2I链路的阴影衰落
        self.delta_distance = []# 距离变化量
        self.V2I_pathloss = [] # V2I链路的路径损耗

        self.vel_v = [] # 车辆速度列表
        self.V2I_channels_abs = []# V2I信道的幅度

        self.V2I_TransmissionRate=[]#车辆到RSU的传输速率(bps)

        self.beta_all = BetaAllocation(self.n_veh) # 实例化BetaAllocation，用于资源分配
        self.RSU_f = 6e9 # RSU的工作频率（Hz）


    def renew_position(self):
        """更新所有车辆的位置，处理车辆转向和出界情况"""
        i = 0
        while (i < len(self.vehicles)):
            # 计算在一个时间间隔t=1s内车辆行驶的距离
            delta_distance = self.vehicles[i].velocity * self.T_n
            change_direction = False # 标记是否改变方向
            # 处理向上行驶的车辆（'u'方向）
            if self.vehicles[i].direction == 'u':

                for j in range(len(self.left_lanes)):
                    # 检查是否到达左车道交叉点，有30%概率左转
                    # 若当前位置的y坐标 + 移动距离 覆盖左车道的y坐标（交叉点）
                    if (self.vehicles[i].start_position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].start_position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.3):
                            # 计算转向后的新位置  调整位置到交叉点，方向改为左（'l'）
                            self.vehicles[i].start_position = [self.vehicles[i].start_position[0] - (
                                    delta_distance - (self.left_lanes[j] - self.vehicles[i].start_position[1])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'# 改变方向为左
                            change_direction = True
                            break
                # 如果未左转，检查是否到达右车道交叉点，有30%概率右转
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].start_position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].start_position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.3):
                                # 计算转向后的新位置
                                self.vehicles[i].start_position = [self.vehicles[i].start_position[0] + (
                                        delta_distance + (self.right_lanes[j] - self.vehicles[i].start_position[1])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'# 改变方向为右
                                change_direction = True
                                break
                # 如果未转向，继续向上行驶
                if change_direction == False:
                    self.vehicles[i].start_position[1] += delta_distance
            # 处理向下行驶的车辆（'d'方向）
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # 检查是否到达左车道交叉点，有30%概率左转
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].start_position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].start_position[1] - delta_distance) <= self.left_lanes[j]):  # come to an crossing
                        if (np.random.uniform(0, 1) < 0.3):
                            # 计算转向后的新位置
                            self.vehicles[i].start_position = [self.vehicles[i].start_position[0] - (
                                        delta_distance - (self.vehicles[i].start_position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l' # 改变方向为左
                            change_direction = True
                            break
                # 如果未左转，检查是否到达右车道交叉点，有30%概率右转
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].start_position[1] >= self.right_lanes[j]) and (
                                self.vehicles[i].start_position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.3):
                                # 计算转向后的新位置
                                self.vehicles[i].start_position = [self.vehicles[i].start_position[0] + (
                                        delta_distance + (self.vehicles[i].start_position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'# 改变方向为右
                                change_direction = True
                                break
                # 如果未转向，继续向下行驶
                if change_direction == False:
                    self.vehicles[i].start_position[1] -= delta_distance
            # 处理向右行驶的车辆（'r'方向）
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # 检查是否到达上行车道交叉点，有30%概率上转
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].start_position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].start_position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.3):
                            # 计算转向后的新位置
                            self.vehicles[i].start_position = [self.up_lanes[j], self.vehicles[i].start_position[1] + (
                                    delta_distance - (self.up_lanes[j] - self.vehicles[i].start_position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u' # 改变方向为上
                            break
                # 如果未上转，检查是否到达下行车道交叉点，有30%概率下转
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].start_position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].start_position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.3):
                                # 计算转向后的新位置
                                self.vehicles[i].start_position = [self.down_lanes[j], self.vehicles[i].start_position[1] - (
                                        delta_distance - (self.down_lanes[j] - self.vehicles[i].start_position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd' # 改变方向为下
                                break
                # 如果未转向，继续向右行驶
                if change_direction == False:
                    self.vehicles[i].start_position[0] += delta_distance
            # 处理向左行驶的车辆（'l'方向）
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                # 检查是否到达上行车道交叉点，有30%概率上转
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].start_position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].start_position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.3):
                            # 计算转向后的新位置
                            self.vehicles[i].start_position = [self.up_lanes[j], self.vehicles[i].start_position[1] + (
                                    delta_distance - (self.vehicles[i].start_position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u' # 改变方向为上
                            break
                # 如果未上转，检查是否到达下行车道交叉点，有30%概率下转
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].start_position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].start_position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.3):
                                # 计算转向后的新位置
                                self.vehicles[i].start_position = [self.down_lanes[j], self.vehicles[i].start_position[1] - (
                                        delta_distance - (self.vehicles[i].start_position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'# 改变方向为下
                                break
                    # 如果未转向，继续向左行驶
                    if change_direction == False:
                        self.vehicles[i].start_position[0] -= delta_distance
            # 处理车辆出界情况（超出环境边界）
            if (self.vehicles[i].start_position[0] < 0) or (self.vehicles[i].start_position[1] < 0) or (
                    self.vehicles[i].start_position[0] > self.width) or (self.vehicles[i].start_position[1] > self.height):
                # 如果车辆向上行驶出界，改变方向为右，并重置位置到右车道
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].start_position = [self.vehicles[i].start_position[0], self.right_lanes[-1]]
                else:
                    # 如果车辆向下行驶出界，改变方向为左，并重置位置到左车道
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].start_position = [self.vehicles[i].start_position[0], self.left_lanes[0]]
                    else:
                        # 如果车辆向左行驶出界，改变方向为上，并重置位置到上车道
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].start_position = [self.up_lanes[0], self.vehicles[i].start_position[1]]
                        else:
                            # 如果车辆向右行驶出界，改变方向为下，并重置位置到下车道
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].start_position = [self.down_lanes[-1], self.vehicles[i].start_position[1]]

            i += 1 # 处理下一辆车

    def add_new_vehicles_by_number(self, n):
        string = 'dulr'
        # 批量添加车辆，每次循环添加4辆不同方向的车（确保方向均匀分布）
        for i in range(int(n/4)):
            # 随机选择车道索引
            ind = np.random.randint(0, len(self.down_lanes))

            # 添加下行方向车辆（'d'）
            # 起始位置：x坐标为选中的下行车道x值，y坐标为2*BS宽度减去10-14之间的随机整数（靠近场景上方）
            start_position = [self.down_lanes[ind], 2*self.BS_width-np.random.randint(10, 15)]
            start_direction = 'd'  # 速度随机10-15m/s   36-54km/h
            vel_v=np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            # 添加上行方向车辆（'u'）
            # 起始位置：x坐标为选中的上行车道x值，y坐标为10-14之间的随机整数（靠近场景下方）
            start_position = [self.up_lanes[ind], np.random.randint(10, 15)]
            start_direction = 'u'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            # 添加左行方向车辆（'l'）
            # 起始位置：x坐标为2*BS宽度减去10-14之间的随机整数（靠近场景右侧），y坐标为选中的左行车道y值
            start_position = [2*self.BS_width-np.random.randint(10, 15), self.left_lanes[ind]]
            start_direction = 'l'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

            # 添加右行方向车辆（'r'）
            # 起始位置：x坐标为10-14之间的随机整数（靠近场景左侧），y坐标为选中的右行车道y值
            start_position = [np.random.randint(10, 15), self.right_lanes[ind]]
            start_direction = 'r'
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))
        # 处理n除以4的余数，补充剩余车辆
        for j in range(int(n % 4)):
            ind = np.random.randint(0, len(self.down_lanes))
            # 随机选择方向
            str = random.choice(string)
            # 初始位置：随机下行车道，y坐标在0到场景高度之间
            start_direction = str
            # 起始位置：x坐标为选中的下行车道x值，y坐标为场景高度范围内的随机值
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            vel_v = np.random.randint(10, 15)
            self.vehicles.append(Vehicle(start_position, start_direction, vel_v))

    def overall_channel_vel_RSU(self):
        """计算车辆与RSU之间的综合信道特性（包含路径损耗和阴影衰落）"""
        # 初始化路径损耗、车辆速度、信道幅度数组
        self.V2I_pathloss = np.zeros((len(self.vehicles)))#车辆速度
        self.vel_v = np.zeros((len(self.vehicles)))#车辆速度
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))#信道幅度
        # 生成阴影衰落（服从均值0、标准差8的正态分布）
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        # 遍历所有车辆计算路径损耗和速度
        for i in range(len(self.vehicles)):  # 计算n辆车的路径损失
            # 计算路径损耗（dB为单位）
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].start_position)
            # 记录车辆速度
            # self.vel_v[i] = self.vehicles[i].velocity
        # 计算综合信道功率（W为单位）：路径损耗转换为线性功率
        self.V2I_overall_W = 1/np.abs(1/np.power(10, self.V2I_pathloss / 10))  # W为单位
        # 综合信道幅度（dB）：包含阴影衰落
        '''self.V2I_overall_W = np.power(10, self.V2I_overall_dB / 10)'''
        self.V2I_channels_abs = 10 * np.log10(self.V2I_overall_W)+self.V2I_Shadowing  #dB

        return self.V2I_channels_abs

    def overall_channel(self,now_timeslot):
        """与overall_channel_vel_RSU功能完全相同，可能为冗余实现"""
        """计算车辆与RSU之间的综合信道特性（包含路径损耗和阴影衰落）"""
        # 初始化路径损耗、车辆速度、信道幅度数组
        self.V2I_pathloss = np.zeros((len(self.vehicles)))# 路径损耗（dB）
        self.vel_v = np.zeros((len(self.vehicles)))# 车辆速度
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))# 综合信道增益（dB）
        # 生成阴影衰落（服从均值0、标准差8dB的正态分布）
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        # 遍历所有车辆计算路径损耗和速度
        for i in range(len(self.vehicles)):  # 计算n辆车的路径损失
            # 计算路径损耗（dB为单位）
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].get_vehicle_location(now_timeslot))
            # 记录车辆速度
            # self.vel_v[i] = self.vehicles[i].velocity
        # 将路径损耗（dB）转换为线性功率衰减因子（W）
        self.V2I_overall_W = 1/np.abs(1/np.power(10, self.V2I_pathloss / 10))  # W为单位
        # 综合路径损耗和阴影衰落，转换为dB单位的信道增益
        self.V2I_channels_abs = 10 * np.log10(self.V2I_overall_W)+self.V2I_Shadowing  #dB

        return self.V2I_channels_abs

    def true_calculate_num(self, action_pf):
        """通过BetaAllocation计算真实任务计算量和标称计算量"""
        # 调用BetaAllocation的true_calculate_times方法
        comp_n_list_true, comp_n_list = self.beta_all.true_calculate_times(action_pf)
        return comp_n_list_true, comp_n_list

    def calculate_num_RSU(self):
        """计算RSU的任务计算能力（单位时间内可处理的任务数）"""
        calculate_num_RSU = self.beta_all.calculate_times_RSU(self.RSU_f)
        return calculate_num_RSU

    def update_buffer(self, comp_n_list):
        """更新车辆的任务缓冲池，累加新生成的任务计算量"""
        for i in range(self.n_veh):
            self.ReplayB_v[i] += comp_n_list[i]

    def trans_energy_RSU(self, action_pf, h_i_dB):
        """计算RSU的传输能量消耗"""
        # 提取功率选择（动作策略中的功率参数）
        p_selection = action_pf[:, 0].reshape(len(self.vehicles), 1)
        # 初始化信号和干扰数组
        V2I_Signals = np.zeros(self.n_veh)
        V2I_Interference = 0
        # 计算每辆车的干扰和接收信号功率
        for i in range(len(self.vehicles)):
            # 累加噪声功率作为干扰（单位：W）
            self.V2I_Interference = V2I_Interference + self.sig2  # 单位：W
            # 计算接收信号功率（W为单位）：考虑功率、路径损耗、天线增益和噪声系数
            V2I_Signals[i] = 10 ** ((p_selection[i] - self.V2I_pathloss_with_fastfading[i]+ self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)  #单位：W
        # 调用BetaAllocation计算传输能量
        V2I_TransmissionRate,trans_energy_RSU = self.beta_all.trans_energy_RSU(action_pf, h_i_dB, self.V2I_Interference)
        # 车辆到RSU的传输速率
        self.V2I_TransmissionRate=V2I_TransmissionRate

        return trans_energy_RSU

    def renew_channel_fastfading(self):
        """更新信道的快衰落特性（小尺度衰落）"""
        # 生成复高斯随机变量（模拟快衰落），计算其幅度的对数（转换为dB）
        self.V2I_pathloss_with_fastfading = 20 * np.log10(
            np.abs(np.random.normal(0, 1, self.V2I_channels_abs.shape) + 1j * np.random.normal(0, 1, self.V2I_channels_abs.shape)) / math.sqrt(2))
        # 结合之前的综合信道幅度，得到包含快衰落的路径损耗
        self.V2I_pathloss_with_fastfading = self.V2I_channels_abs - self.V2I_pathloss_with_fastfading

    def RSU_reward1(self, action_pf, comp_n_list_true, trans_energy_RSU, offload_num):
        """计算RSU的奖励值，考虑能量消耗、任务卸载效率等因素"""
        # 初始化总能量数组、RSU计算能量、车辆本地计算能量
        E_total = np.zeros(self.n_veh)
        RSU_energy = self.beta_all.energy_RSU(self.RSU_f)# RSU单位计算能量
        E_single_energy_vel = self.beta_all.single_comp_energy_vel(action_pf) # 车辆本地计算能量
        Delay_vel=np.zeros(self.n_veh)

        cf = 0# 过载惩罚系数
        overload = []# 过载任务量（卸载量超过缓冲池任务的部分）
        load_rate_0 = []# 实际卸载量记录
        ReplayB_v_copy = list(self.ReplayB_v)# 备份当前缓冲池任务量
        # 遍历每辆车计算能量消耗、时延    进行缓冲池更新
        for i in range(self.n_veh):
            uu=1# 传输能量系数（若卸载量则为0）
            # 情况1：缓冲池任务量 > 卸载任务量
            if self.ReplayB_v[i]>offload_num[i]:
                if offload_num[i] ==0:
                    uu=0# 无卸载时传输能量为0
                load_rate_0.append(offload_num[i])# 记录实际卸载量
                self.ReplayB_v[i] -= offload_num[i] # 缓冲池减去卸载量
                # 子情况1.1：剩余缓冲任务 > 本地可计算量
                if self.ReplayB_v[i]>comp_n_list_true[i]:
                    # 能量 = 本地计算能量 + 传输能量 + RSU计算能量
                    E_total[i] = comp_n_list_true[i]*E_single_energy_vel[i]+ uu*trans_energy_RSU[i] + offload_num[i]* RSU_energy
                    # 时延 = max(本地计算时间，卸载传输时间+RSU计算时间)  若offload_num[i]=0则不卸载，时延=本地计算时间
                    # 卸载传输时间=(卸载的数据Z+模型数据D)
                    #卸载任务circle/处理单位大小的数据所需CPU周期数=卸载任务数据量（单位KB）
                    trans_delay=(offload_num[i]/self.beta_all.r_n+self.beta_all.D_n)/(self.V2I_TransmissionRate[i]/8)
                    caculate_local=comp_n_list_true[i]/action_pf[i][1]
                    caculate_rsu=offload_num[i]/self.RSU_f
                    Delay_vel[i] =max(caculate_local,trans_delay+caculate_rsu)

                    self.ReplayB_v[i] -= comp_n_list_true[i]
                # 子情况1.2：剩余缓冲任务 <= 本地可计算量
                else:
                    # 能量 = RSU计算能量 + 传输能量 + 剩余任务本地计算能量
                    E_total[i] = offload_num[i]* RSU_energy + uu*trans_energy_RSU[i] + self.ReplayB_v[i]*E_single_energy_vel[i]

                    # 时延
                    trans_delay = (offload_num[i] / self.beta_all.r_n + self.beta_all.D_n) / (self.V2I_TransmissionRate[i] / 8)
                    caculate_local = self.ReplayB_v[i] / action_pf[i][1]
                    caculate_rsu = offload_num[i] / self.RSU_f
                    Delay_vel[i] = max(caculate_local, trans_delay + caculate_rsu)

                    self.ReplayB_v[i] = 0# 缓冲池清空
            # 情况2：缓冲池任务量 <= 卸载任务量（出现过载）
            else:
                cf = 0.001 # 激活过载惩罚

                load_rate_0.append(self.ReplayB_v[i]) # 记录实际卸载量（等于缓冲池任务量）
                overload.append(offload_num[i] - self.ReplayB_v[i])# 计算过载量
                offload_num[i] = self.ReplayB_v[i]# 修正卸载量为缓冲池任务量
                # 能量 = RSU计算能量 + 传输能量
                E_total[i] = offload_num[i] * RSU_energy + uu*trans_energy_RSU[i]

                # 时延 没有本地计算
                trans_delay = (offload_num[i] / self.beta_all.r_n + self.beta_all.D_n) / (
                            self.V2I_TransmissionRate[i] / 8)
                caculate_rsu = offload_num[i] / self.RSU_f
                Delay_vel[i] = trans_delay + caculate_rsu

                self.ReplayB_v[i] = 0 # 缓冲池清空
        # 计算任务卸载率（实际卸载量/初始缓冲量）
        array1 = np.array(load_rate_0)
        array2 = np.array(ReplayB_v_copy)
        rate_0_temp = []
        for i in range(len(array1)):
            if array2[i] == 0:
                continue # 避免除以0
            rate_0_temp.append(array1[i] / array2[i])

        rate_0 = [x for x in rate_0_temp if x != 0]# 过滤无效值


        # 新增：将时延纳入奖励（惩罚大时延）
        delay_penalty =4000 * sum(Delay_vel)  # 时延惩罚系数，可调整
        #源代码 不考虑时延
        # reward_tot = 10 * sum(E_total) + cf * sum(overload) + 0.01 * sum(self.ReplayB_v)
        # 总奖励计算：能量消耗惩罚 + 过载惩罚 + 剩余缓冲任务惩罚       +时延
        # reward_tot = 10*sum(E_total) + cf * sum(overload) + 0.01 * sum(self.ReplayB_v)+delay_penalty

        reward_tot = 10 * sum(E_total) + 0.01 * sum(self.ReplayB_v) + delay_penalty

        # # 新增：将时延纳入奖励（惩罚大时延）
        # delay_penalty = 1 * sum(Delay_vel)  # 时延惩罚系数，可调整
        # # 总奖励计算：能量消耗惩罚 + 过载惩罚 + 剩余缓冲任务惩罚       +时延
        # # reward_tot = (10 * sum(E_total) + cf * sum(overload) + 0.01 * sum(self.ReplayB_v))*math.sqrt(delay_penalty)#开方
        # reward_tot = (10 * sum(E_total) + cf * sum(overload) + 0.01 * sum(self.ReplayB_v)) * delay_penalty#不开方

        return E_total, reward_tot, sum(overload), rate_0,Delay_vel


    def get_state(self):
        """获取环境状态特征（用于强化学习）"""
        # 归一化信道幅度（除以总信道幅度）
        bb = sum(self.V2I_channels_abs) # 计算所有车辆的信道幅度总和
        V2I_abs= self.V2I_channels_abs/bb# 每个车辆的信道幅度除以总和，实现归一化
        # 归一化车辆速度（除以20）
        vel_v = self.vel_v /20 # 每个车辆的速度除以20，将速度范围映射到 [0, 1] 附近
        # 拼接信道和速度特征作为状态
        return np.concatenate((np.reshape(V2I_abs, -1), np.reshape(vel_v, -1)))


    def Compute_Performance_Reward_Train(self, action_pf, h_i_dB, vel_v, lambda_1, lambda_2):
        """训练时计算性能指标和奖励，基于资源分配策略"""
        # 提取功率选择
        p_selection = action_pf[:, 0].reshape(len(self.vehicles), 1)
        V2I_Signals = np.zeros(self.n_veh)
        # 初始化干扰为噪声功率
        V2I_Interference = self.sig2 # 单位：W
        # 计算每辆车的接收信号功率
        for i in range(len(self.vehicles)):
            V2I_Signals[i] = 10 ** ((p_selection[i] - self.V2I_channels_abs[i]+ self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        # 调用BetaAllocation进行资源分配并返回性能指标
        E_total, tran_success, c_total, reward, comp_n_list = self.beta_all.beta_allocation(action_pf, h_i_dB, vel_v, lambda_1, lambda_2, V2I_Interference)

        return E_total, tran_success, c_total, reward, comp_n_list


    def act_for_training(self, action_pf, h_i_dB, vel_v, lambda_1, lambda_2):
        """训练时执行动作的封装方法，调用Compute_Performance_Reward_Train"""
        E_total, tran_success, c_total, reward, comp_n_list = self.Compute_Performance_Reward_Train(action_pf, h_i_dB, vel_v, lambda_1, lambda_2)

        return E_total, tran_success, c_total, reward, comp_n_list

    def init_time_slots(self):
        self.time_slots: dataStruct.timeSlots = dataStruct.timeSlots(
            start=time_slot_start,
            end=time_slot_end,
            slot_length=time_slot_length,
        )
    def generate_vehicles_by_number(self):

        self.vehicle_list: dataStruct.vehicleList = dataStruct.vehicleList(
            edge_number=edge_number,
            communication_range=communication_range,
            vehicle_number=self.n_veh,
            time_slots=self.time_slots,
            trajectories_file_name=trajectories_file_name,
            slot_number=time_slot_number,
            task_number=task_number,
            task_request_rate=task_request_rate,
            seeds=vehicle_seeds,
        )
        self.vehicles=self.vehicle_list.get_vehicle_list()
    def new_random_game(self, n_veh=0):
        """初始化新的模拟场景"""
        # 清空车辆列表和干扰车辆列表
        self.vehicles = []
        self.vehicles_interference = []
        # 若指定车辆数，则更新车辆数
        if n_veh > 0:
            self.n_veh = n_veh
        # 添加指定数量的新车辆
        # self.add_new_vehicles_by_number(self.n_veh)

        #初始化时间隙
        self.init_time_slots()

        #根据CSV文件生成车辆
        self.generate_vehicles_by_number()

        # 计算信道特性
        self.overall_channel(0)#初始化时 时隙为0
        # 更新快衰落
        self.renew_channel_fastfading()
        # 初始化任务缓冲池为0
        self.ReplayB_v = np.zeros(self.n_veh)
        #
        self.V2I_TransmissionRate=[]
