import math
import numpy as np
import random


class BetaAllocation:
    def __init__(self, n_veh):
        # 初始化类实例，参数n_veh为车辆数量

        # 全局变量
        self.n_veh = n_veh# 车辆数量

        # 以下为各类系统参数初始化
        self.Z = 11.2# 模型大小（单位：MB）
        self.B = 2000000# 总带宽（单位：Hz）
        # 计算路径损耗（单位：dB），公式基于自由空间传播模型，参数包括距离、频率等
        self.h_n_dB = 20 * math.log10((4 * math.pi * 200 * 915e6) / 3e8)
        self.I_n = 0# 干扰功率（初始化为0）
        self.N_0 = 10 ** (-114 / 10)# 噪声功率谱密度（单位：W/Hz），由-114dBm/Hz转换而来
        self.T_n = 1 # 时间周期（单位：s）
        self.data_t = 0.02 # 数据传输时间（单位：s）
        self.k = 1e-27# 能量系数（与计算能耗相关）
        self.r_n = 1600# cycle/B 为处理单位大小的数据所需CPU周期数
        self.D_n = 1500# 计算任务的数据量（单位：KB）
        self.q_tao = 0.2  # 可能为时间占空比参数
        self.bsAntGain = 8 # 基站天线增益（单位：dBi）
        self.bsNoiseFigure = 5# 基站噪声系数（单位：dB）
        self.vehAntGain = 3# 车辆天线增益（单位：dBi）
        self.vehNoiseFigure = 11 # 车辆噪声系数（单位：dB）
        self.m = 0.023# 可能为移动性相关参数

    def true_calculate_times(self, p_f):
        # 计算每个车辆的真实计算次数和理论计算次数
        # p_f为包含功率和频率的列表，每个元素为[功率, cpu频率，rsu分配cpu比例]
        list_num_true = [] # 存储真实计算次数
        list_num = []# 存储理论计算次数
        for i in range(self.n_veh):
            # 随机生成真实时间比例（0.05到1之间），乘以周期得到真实可用时间
            random_value = random.uniform(0.05,1)
            T_true = random_value * self.T_n # 真实可用时间（单位：s）
            # 计算单次(circle)计算所需时间：总计算量（D_n * r_n）除以频率（p_f[i][1]）
            T_comp = self.D_n * self.r_n /p_f[i][1]

            # 理论计算次数：(总周期时间 - 数据传输时间) / 单次计算时间，取整
            num =round((self.T_n-self.data_t)/T_comp)
            # 真实计算次数：(真实可用时间 - 数据传输时间) / 单次计算时间，取整
            num_true = round((T_true-self.data_t)/T_comp)

            list_num_true.append(num_true)
            list_num.append(num)
        return list_num_true, list_num # 返回真实计算次数列表和理论计算次数列表


    def calculate_times_RSU(self, RSU_f):
        # 计算RSU（路侧单元）的计算次数
        # RSU_f为RSU的计算频率
        # 公式：(总周期时间 - 数据传输时间) / (总计算量 / RSU频率)，取整
        calculate_times_RSU = round((self.T_n-self.data_t)/(self.D_n * self.r_n /RSU_f))
        return calculate_times_RSU#返回RSU的计算次数

    def energy_RSU(self, RSU_f):
        # 计算RSU的计算能耗
        # 公式：能量系数 * 计算复杂度 * 数据量 * 频率的平方（经典的计算能耗模型）
        E = self.k * self.D_n * self.r_n * RSU_f**2
        return E

    def trans_energy_RSU(self, p_f, h_i_dB, V2I_Interference):
        # 计算车辆到基础设施（V2I）的传输能耗
        # p_f为车辆的功率和频率列表，h_i_dB为路径损耗（dB），V2I_Interference为V2I干扰功率
        V2I_Signals_dB = np.zeros(self.n_veh)# 存储每个车辆的接收信号强度（dB）
        V2I_Signals_W = np.zeros(self.n_veh)# 存储每个车辆的接收信号强度（W）
        V2I_TransmissionRate=np.zeros(self.n_veh)#存储每个车辆到RSU的传输速率（）

        trans_energy_RSU = np.zeros(self.n_veh) # 存储每个车辆的传输能耗
        count = 0# 计数器，用于索引车辆
        for p_f_1 in p_f:
            # 计算接收信号强度（dB）：发射功率 - 路径损耗 + 车辆天线增益 + 基站天线增益 - 基站噪声系数
            V2I_Signals_dB[count] = p_f_1[0] - h_i_dB[count] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
            # 将dB转换为瓦特（W）
            V2I_Signals_W[count] = 10 ** (
                        (p_f_1[0] - h_i_dB[count] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
            # 计算传输能耗：发射功率 * (模型数据量 + 任务数据量)  / 传输速率
            # 其中信噪比为接收信号强度除以干扰功率
            #源码：self.Z+self.D_n D=11.2MB（模型大小）+1500KB  单位没有进行换算  好像也没有考虑卸载了多少数据，直接算所有数据的卸载能耗？ 而且不知道为什么要/车辆数量：B是总带宽，系统采用等带宽分配，所有车辆平均分配带宽
            # trans_energy_RSU[count] = p_f_1[0]*(self.Z+self.D_n) / self.B/len(trans_energy_RSU) / math.log(1 + V2I_Signals_W[count] / V2I_Interference, math.e)

            #每辆车传输速率（bps）
            V2I_TransmissionRate[count]=self.B/ len(trans_energy_RSU)*math.log(1 + V2I_Signals_W[count] / V2I_Interference, math.e)
            trans_energy_RSU[count] = p_f_1[0]*(self.Z*1000+self.D_n) / V2I_TransmissionRate[count]

            count += 1# 计数器递增
        return V2I_TransmissionRate,trans_energy_RSU# 返回每个车辆的传输能耗列表  顺便返回车辆到RSU的传输速率


    def single_comp_energy_vel(self,pf):
        # 计算单个车辆的计算能耗
        # pf为包含功率和频率的列表，每个元素为[功率, 频率]
        E_list = [] # 存储每个车辆的计算能耗
        for i in range(self.n_veh):
            # 公式：能量系数 * 计算复杂度 * 数据量 * 频率的平方（与RSU计算能耗模型一致）
            E = self.k * self.D_n * self.r_n * pf[i][1] ** 2
            E_list.append(E) # 添加到能耗列表
        return E_list # 返回每个车辆的计算能耗列表


    # 时延计算
    def compute_transmission_delay(self, r_n, B, P, G, N0,dis):#r_n任务数据量,B带宽,P传输功率,G信道增益,N0单位为W的噪声功率,dis车辆与RSU的距离
        """计算通信时延（单位：s）"""
        SNR = (P * G*dis) / (N0)  # 信噪比  (传输功率*信道增益*车辆与RSU的距离)/噪声功率
        if SNR <= 0:
            return float('inf')  # 信道质量过差，时延无穷大
        rate = B * np.log2(1 + SNR)  # 传输速率（bit/s）
        return r_n / rate  # 通信时延

    def compute_local_delay(self, C, f_veh):
        """计算本地计算时延（单位：s）"""
        if f_veh <= 0:
            return float('inf')
        return C / f_veh  # C为任务计算量（CPU周期）

    def compute_remote_delay(self, C, f_rsu):
        """计算RSU计算时延（单位：s）"""
        if f_rsu <= 0:
            return float('inf')
        return C / f_rsu


    