import matplotlib.pyplot as plt
import numpy as np
import random

# 从main_train.py中提取车道参数
up_lanes = [200.875, 202.625, 400.875, 402.625]
down_lanes = [197.375, 199.125, 397.375, 399.125]
left_lanes = [200.875, 202.625, 400.875, 402.625]
right_lanes = [197.375, 199.125, 397.375, 399.125]

# 场景尺寸 (来自main_train.py)
width = 1000
height = 1000
lane_width = 3.5  # 车道宽度（代码中隐含的3.5米）


def add_new_vehicles_by_number(n_veh):
    """模拟车辆生成逻辑，在各车道随机生成车辆"""
    vehicles = []
    directions = ['u', 'd', 'l', 'r']  # 上、下、左、右

    for _ in range(n_veh):
        # 随机选择行驶方向
        direction = random.choice(directions)

        if direction == 'u':  # 上行车道（沿y轴正方向）
            # 选择一条上行车道，y坐标在车道范围内，x坐标在场景内随机
            lane_y = random.choice(up_lanes)
            y = random.uniform(lane_y - lane_width / 2, lane_y + lane_width / 2)
            x = random.uniform(0, width)

        elif direction == 'd':  # 下行车道（沿y轴负方向）
            lane_y = random.choice(down_lanes)
            y = random.uniform(lane_y - lane_width / 2, lane_y + lane_width / 2)
            x = random.uniform(0, width)

        elif direction == 'l':  # 左车道（沿x轴负方向）
            lane_x = random.choice(left_lanes)
            x = random.uniform(lane_x - lane_width / 2, lane_x + lane_width / 2)
            y = random.uniform(0, height)

        else:  # 右车道（沿x轴正方向）
            lane_x = random.choice(right_lanes)
            x = random.uniform(lane_x - lane_width / 2, lane_x + lane_width / 2)
            y = random.uniform(0, height)

        vehicles.append((x, y, direction))

    return vehicles


def plot_environment(vehicles):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # SimHei 优先，英文字体备用
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    """绘制环境模型和车辆位置"""
    plt.figure(figsize=(10, 10))
    plt.xlim(0, width)
    plt.ylim(0, height)

    # 绘制上行车道（水平车道，y为中心）
    for y in up_lanes:
        plt.fill_between(
            [0, width],
            y - lane_width / 2,
            y + lane_width / 2,
            color='lightgreen',
            alpha=0.5,
            label='上行车道' if y == up_lanes[0] else ""
        )
        plt.text(width + 10, y, f'y={y}', verticalalignment='center')

    # 绘制下行车道
    for y in down_lanes:
        plt.fill_between(
            [0, width],
            y - lane_width / 2,
            y + lane_width / 2,
            color='lightblue',
            alpha=0.5,
            label='下行车道' if y == down_lanes[0] else ""
        )
        plt.text(width + 10, y, f'y={y}', verticalalignment='center')

    # 绘制左车道（垂直车道，x为中心）
    for x in left_lanes:
        plt.fill_betweenx(
            [0, height],
            x - lane_width / 2,
            x + lane_width / 2,
            color='lightcoral',
            alpha=0.5,
            label='左车道' if x == left_lanes[0] else ""
        )
        plt.text(x, height + 10, f'x={x}', horizontalalignment='center')

    # 绘制右车道
    for x in right_lanes:
        plt.fill_betweenx(
            [0, height],
            x - lane_width / 2,
            x + lane_width / 2,
            color='lightsalmon',
            alpha=0.5,
            label='右车道' if x == right_lanes[0] else ""
        )
        plt.text(x, height + 10, f'x={x}', horizontalalignment='center')

    # 绘制车辆
    dir_marker = {'u': '↑', 'd': '↓', 'l': '←', 'r': '→'}
    for x, y, d in vehicles:
        plt.scatter(x, y, s=50, c='black', zorder=3)
        plt.text(x + 5, y + 5, dir_marker[d], fontsize=12)

    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('车道环境模型与随机车辆位置')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


# 生成10辆随机车辆并绘制
if __name__ == "__main__":
    vehicles = add_new_vehicles_by_number(10)
    plot_environment(vehicles)