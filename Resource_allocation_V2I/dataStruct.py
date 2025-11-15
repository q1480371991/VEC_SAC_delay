import sys

import numpy as np
import pandas as pd
from typing import List

import random


class timeSlots(object):
    """系统的离散时间槽集合"""

    def __init__(
            self,
            start: int,
            end: int,
            slot_length: int) -> None:
        """初始化时间槽
                Args:
                    start: 系统开始时间
                    end: 系统结束时间
                    slot_length: 每个时间槽的长度（单位时间）
        """
        self._start = start  # 开始时间
        self._end = end  # 结束时间
        self._slot_length = slot_length  # 单个时间槽长度
        self._number = int((end - start + 1) / slot_length)  # 计算总时间槽数量：(结束时间-开始时间+1)/每个时间槽长度
        self._now = start  # 当前时间，初始化为开始时间
        self.reset()  # 重置时间状态

    def __str__(self) -> str:
        """返回时间槽的字符串表示"""
        return f"now time: {self._now}, [{self._start} , {self._end}] with {self._slot_length} = {self._number} slots"

    def add_time(self) -> None:
        """将系统时间向前推进1单位"""
        self._now += 1

    def is_end(self) -> bool:
        """检查系统是否到达时间槽末尾
                Returns:
                    若当前时间 >= 结束时间则返回True，否则返回False
        """
        return self._now >= self._end

    def get_slot_length(self) -> int:
        """获取每个时间槽的长度
                Returns:
                    时间槽长度
        """
        return int(self._slot_length)

    def get_number(self) -> int:
        """获取总时间槽数量
               Returns:
                   时间槽总数
        """
        return int(self._number)

    def now(self) -> int:
        """获取当前时间
                Returns:
                    当前时间值
        """
        return int(self._now)

    def get_start(self) -> int:
        """获取开始时间
                Returns:
                    开始时间值
        """
        return int(self._start)

    def get_end(self) -> int:
        """获取结束时间
                Returns:
                    结束时间值
        """
        return int(self._end)

    def reset(self) -> None:
        """重置当前时间为开始时间"""
        self._now = self._start


class task(object):
    """任务实体类，存储单个任务的属性"""
    def __init__(self, task_index: int, data_size: float, computation_cycles: float, delay_threshold: float) -> None:
        self._task_index = task_index# 任务索引（唯一标识）
        self._data_size = data_size# 任务数据量（单位：）
        self._computation_cycles = computation_cycles # 任务所需计算周期（单位：如CPU周期）
        self._delay_threshold = delay_threshold # 任务延迟阈值（超过此值任务失败）
    def get_task_index(self) -> int:
        """获取任务索引
               Returns:
                   任务索引值
        """
        return int(self._task_index)
    def get_data_size(self) -> float:
        """获取任务数据量
                Returns:
                    数据量大小
        """
        return float(self._data_size)
    def get_computation_cycles(self) -> float:
        """获取任务所需计算周期
                Returns:
                    计算周期数
        """
        return float(self._computation_cycles)
    def get_delay_threshold(self) -> float:
        """获取任务延迟阈值
                Returns:
                    延迟阈值
        """
        return float(self._delay_threshold)


class location(object):
    """节点位置类，存储x、y坐标及距离计算方法"""
    def __init__(self, x: float, y: float) -> None:
        """初始化位置
        Args:
            x: x坐标
            y: y坐标
        """
        self._x = x# x坐标
        self._y = y# y坐标

    def __str__(self) -> str:
        """返回位置的字符串表示"""
        return f"x: {self._x}, y: {self._y}"
    def get_x(self) -> float:
        return self._x
    def get_y(self) -> float:
        return self._y
    def get_distance(self, location: "location") -> float:
        """计算与另一个位置的欧氏距离
                Args:
                    location: 目标位置对象
                Returns:
                    两点之间的距离
        """
        return np.sqrt(
            (self._x - location.get_x())**2 +
            (self._y - location.get_y())**2
        )


class trajectory(object):
    """节点轨迹类，存储不同时间槽的位置信息"""
    def __init__(self, timeSlots: timeSlots, locations: List[location]) -> None:
        """初始化轨迹
                Args:
                    timeSlots: 时间槽列表
                    locations: 位置列表，每个元素对应一个时间槽的位置
        """

        self._locations = locations# 位置列表：index对应时间槽索引

        # 注释：原代码有轨迹长度与时间槽数量的校验，此处可能因需求注释掉
        # if len(self._locations) != timeSlots.get_number():
        #     raise ValueError("The number of locations must be equal to the max_timestampes.")

    def __str__(self) -> str:
        """返回轨迹的字符串表示（所有位置信息）"""
        return str([str(location) for location in self._locations])

    def get_location(self, nowTimeSlot: int) -> location:
        """获取指定时间槽的位置
                Args:
                    nowTimeSlot: 时间槽索引
                Returns:
                    对应时间槽的location对象；若索引越界返回None
        """
        try:
            return self._locations[nowTimeSlot]
        except IndexError:
            return None

    def get_locations(self) -> List[location]:
        """获取所有时间槽的位置列表
        Returns:
            包含所有location对象的列表
        """
        return self._locations

    def __str__(self) -> str:
        """打印轨迹详情（按时间槽索引排列）"""
        """ print the trajectory.
        Returns:
            the string of the trajectory.
        """
        print_result= ""
        for index, location in enumerate(self._locations):
            if index % 10 == 0:
                print_result += "\n"
            print_result += "(" + str(index) + ", "
            print_result += str(location.get_x()) + ", "
            print_result += str(location.get_y()) + ")"
        return print_result


class vehicle(object):
    """"车辆节点类，存储单辆车的属性、轨迹及任务请求信息"""

    def __init__(
            self,
            vehicle_index: int,
            vehicle_trajectory: trajectory,
            slot_number: int,
            task_number: int,
            task_request_rate: float,
            seed: int,
    ) -> None:
        self._vehicle_index = vehicle_index  # 车辆唯一索引
        self._vehicle_trajectory = vehicle_trajectory  # 车辆轨迹对象（包含各时间槽的位置）
        self._slot_number = slot_number  # 总时间槽数量
        self._task_number = task_number  # 总任务数量
        self._task_request_rate = task_request_rate  # 任务请求概率（每个时间槽请求任务的概率）
        self._seed = seed  # 随机种子（保证任务请求的可复现性）
        # 初始化时生成车辆在各时间槽的任务请求列表
        self._requested_tasks = self.tasks_requested()

    def get_vehicle_index(self) -> int:
        return int(self._vehicle_index)

    def get_requested_tasks(self) -> List[int]:
        """获取所有时间槽的任务请求列表
        Returns:
            列表元素为任务索引（-1表示该时间槽无请求）
        """
        return self._requested_tasks

    def get_requested_task_by_slot_index(self, slot_index: int) -> int:
        """获取指定时间槽的请求任务
                Args:
                    slot_index: 时间槽索引
                Returns:
                    任务索引（-1表示无请求）
        """
        return self._requested_tasks[slot_index]

    def get_vehicle_location(self, nowTimeSlot: int) -> location:
        """获取车辆在指定时间槽的位置
                Args:
                    nowTimeSlot: 时间槽索引
                Returns:
                    位置对象（location）
        """
        return self._vehicle_trajectory.get_location(nowTimeSlot)

    def get_distance_between_edge(self, nowTimeSlot: int, edge_location: location) -> float:
        """计算车辆在指定时间槽与边缘节点的距离
                Args:
                    nowTimeSlot: 时间槽索引
                    edge_location: 边缘节点位置对象
                Returns:
                    欧氏距离值
        """
        return self._vehicle_trajectory.get_location(nowTimeSlot).get_distance(edge_location)

    def get_vehicle_trajectory(self) -> trajectory:
        return self._vehicle_trajectory

    def tasks_requested(self) -> List[int]:
        """生成车辆在各时间槽的任务请求计划
                Returns:
                    长度为总时间槽数的列表，元素为任务索引（-1表示无请求）
        """
        # 计算总请求任务数：总时间槽数 × 请求概率
        requested_task_number = int(self._slot_number * self._task_request_rate)
        # 初始化请求列表，-1表示该时间槽无任务请求
        requested_tasks = np.zeros(self._slot_number)
        # 确定任务请求的时间槽索引
        for index in range(self._slot_number):
            requested_tasks[index] = -1
        if requested_task_number == self._slot_number:
            # 若请求概率为100%，则所有时间槽都有请求
            task_requested_time_slot_index = list(range(self._slot_number))
        else:
            # 随机选择指定数量的时间槽（无重复）
            np.random.seed(self._seed)
            task_requested_time_slot_index = list(
                np.random.choice(self._slot_number, requested_task_number, replace=False))
        # 为每个请求时间槽随机分配任务索引（可重复）
        np.random.seed(self._seed)
        task_requested_task_index = list(np.random.choice(self._task_number, requested_task_number, replace=True))
        # 填充任务请求列表
        for i in range(len(task_requested_time_slot_index)):
            requested_tasks[task_requested_time_slot_index[i]] = task_requested_task_index[i]
        return requested_tasks


class vehicleList(object):
    """车辆列表类，管理多辆车的集合及轨迹数据读取"""

    def __init__(
            self,
            edge_number: int,  # 边缘节点总数
            communication_range: float,  # 边缘节点通信范围
            vehicle_number: int,  # 车辆总数
            time_slots: timeSlots,  # 时间槽列表
            trajectories_file_name: str,  # 轨迹数据文件前缀
            slot_number: int,  # 总时间槽数
            task_number: int,  # 总任务数
            task_request_rate: float,  # 任务请求概率
            seeds: List[int]  # 随机种子列表（每辆车一个）
    ) -> None:
        self._edge_number = edge_number
        self._communication_range = communication_range
        self._vehicle_number = vehicle_number
        # 每个边缘节点覆盖的车辆数（平均分配）
        self._vehicle_number_in_edge = int(self._vehicle_number / self._edge_number)
        self._trajectories_file_name = trajectories_file_name  # 轨迹文件路径前缀
        self._slot_number = slot_number
        self._task_number = task_number
        self._task_request_rate = task_request_rate
        self._seeds = seeds  # 每辆车对应一个随机种子
        # 读取并初始化所有车辆的轨迹
        self._vehicle_trajectories = self.read_vehicle_trajectories(time_slots)
        self._vehicle_list = [
            vehicle(
                vehicle_index=vehicle_index,
                vehicle_trajectory=vehicle_trajectory,
                slot_number=self._slot_number,
                task_number=self._task_number,
                task_request_rate=self._task_request_rate,
                seed=seed)
            for vehicle_index, vehicle_trajectory, seed in zip(
                range(self._vehicle_number), self._vehicle_trajectories, self._seeds)
        ]

    def get_vehicle_number(self) -> int:
        return int(self._vehicle_number)

    def get_slot_number(self) -> int:
        return int(self._slot_number)

    def get_task_number(self) -> int:
        return int(self._task_number)

    def get_task_request_rate(self) -> float:
        return float(self._task_request_rate)

    def get_vehicle_list(self) -> List[vehicle]:
        return self._vehicle_list

    def get_vehicle_by_index(self, vehicle_index: int) -> vehicle:
        return self._vehicle_list[int(vehicle_index)]

    def read_vehicle_trajectories(self, timeSlots: timeSlots) -> List[trajectory]:
        """读取并解析车辆轨迹文件，生成轨迹对象列表
                Args:
                    timeSlots: 时间槽
                Returns:
                    包含所有车辆轨迹的列表
        """
        # 边缘节点在宽度方向的数量（边缘节点呈正方形网格分布）
        edge_number_in_width = int(np.sqrt(self._edge_number))
        vehicle_trajectories: List[trajectory] = []
        # 遍历每个边缘节点所在的网格坐标(i,j)
        # i和j分别代表网格的行和列索引
        # 轨迹文件路径：前缀 + 网格坐标（如"trajectories_0_1.csv"）
        trajectories_file_name = self._trajectories_file_name + '.csv'
        # 读取CSV文件，列名为车辆ID、时间、经度、纬度
        df = pd.read_csv(
            trajectories_file_name,
            names=['vehicle_id', 'time', 'longitude', 'latitude'], header=0)
        # 获取文件中最大的车辆ID（用于遍历所有车辆）
        max_vehicle_id = df['vehicle_id'].max()
        # 筛选在边缘节点通信范围内的车辆ID
        selected_vehicle_id = []
        for vehicle_id in range(int(max_vehicle_id)):
            # 提取该车辆的所有轨迹数据
            new_df = df[df['vehicle_id'] == vehicle_id]
            # 计算该车辆活动范围的边界坐标
            max_x = new_df['longitude'].max()
            max_y = new_df['latitude'].max()
            min_x = new_df['longitude'].min()
            min_y = new_df['latitude'].min()
            # 计算车辆活动范围到边缘节点中心的最远距离和最近距离
            # （假设边缘节点中心在通信范围的中心位置）
            max_distance = np.sqrt(
                (max_x - self._communication_range) ** 2 + (max_y - self._communication_range) ** 2)
            min_distance = np.sqrt(
                (min_x - self._communication_range) ** 2 + (min_y - self._communication_range) ** 2)
            # 若车辆的所有位置都在边缘节点的通信范围内，则选中该车辆
            # （最远距离 < 通信范围，确保完全覆盖）
            if max_distance < self._communication_range and min_distance < self._communication_range:
                selected_vehicle_id.append(vehicle_id)
        # 检查选中的车辆数量是否满足需求
        if len(selected_vehicle_id) < self._vehicle_number_in_edge:
            raise ValueError(
                f' len(selected_vehicle_id): {len(selected_vehicle_id)} Error: vehicle number in edge is less than expected')
        # 为选中的车辆创建轨迹对象（取前N辆，N=每个边缘节点分配的车辆数）
        # 原来的顺序选择代码
        # for vehicle_id in selected_vehicle_id[: self._vehicle_number_in_edge]:

        # 修改为随机选择
        for vehicle_id in random.sample(selected_vehicle_id, self._vehicle_number_in_edge):
            # 提取该车辆的轨迹数据
            new_df = df[df['vehicle_id'] == vehicle_id]
            # 存储该车辆在所有时间点的位置列表
            loc_list: List[location] = []
            # 提取该车辆每个时间点的位置并转换坐标
            for row in new_df.itertuples():
                # time = getattr(row, 'time')

                # 将局部坐标转换为全局坐标
                # 网格偏移规则：第i行j列的网格，x方向偏移i*2*通信范围，y方向偏移j*2*通信范围
                # 确保不同网格的坐标不重叠
                x = getattr(row, 'longitude')
                y = getattr(row, 'latitude')
                loc = location(x, y)
                loc_list.append(loc)
            # 创建轨迹对象
            new_vehicle_trajectory: trajectory = trajectory(
                timeSlots=timeSlots,
                locations=loc_list
            )
            vehicle_trajectories.append(new_vehicle_trajectory)


        return vehicle_trajectories