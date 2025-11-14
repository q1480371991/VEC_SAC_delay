import numpy as np
import pandas as pd
from typing import List


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
        return np.math.sqrt(
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