import pickle
import matplotlib.pyplot as plt
import os

import numpy as np


def draw():
    # 创建保存目录
    log_dir = 'compare_plot_log'
    os.makedirs(log_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ======== 1. 对比的实验 ========
    runs = {
        "源码": "./log/SAC_0_orogin_episode1000_nodelay/SAC_data_0.pkl",
        "不考虑时延": "./log/SAC_0_episode1000_nodelay/SAC_data_0.pkl",
        "考虑时延": "./log/SAC_0_episode1000_delay4000/SAC_data_0.pkl",
    }

    # ======== 2. 指标及其对应的 key（注意这里用的是你自己定义的 Sum_xxx） ========
    metric_keys = {
        "奖励 (reward)": "Sum_reward",
        "时延 (delay)": "Sum_delay",
        "能耗 (E_total)": "Sum_E_total",
        "卸载率 (load_rate_0)": "Sum_load_rate_0",
        "过载率 (eta1)": "Sum_eta1"
    }

    # ======== 3. 读取所有 pkl ========
    # 读取所有数据
    results = {}
    min_length = float('inf')  # 找到最小数据长度

    # 第一遍：读取数据并找到最小长度
    for name, path in runs.items():
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        # 找到第一个数组键的长度作为参考
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                min_length = min(min_length, len(value))
                break

    print(f"统一数据长度为: {min_length} episodes")

    # 第二遍：统一截取数据
    for name, path in runs.items():
        if not os.path.exists(path):
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        # 统一截取所有数据
        unified_data = {}
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > min_length:
                unified_data[key] = value[:min_length]
            else:
                unified_data[key] = value

        results[name] = unified_data

    # ======== 4. 绘图并保存 ========
    for zh_name, key in metric_keys.items():
        plt.figure()

        for name, data in results.items():
            if key not in data:
                print(f"[警告] {name} 中不存在 key {key}，跳过该方法")
                continue

            y = data[key]
            x = range(len(y))
            plt.plot(x, y, label=name)

        plt.xlabel("训练轮次 / Episode")
        plt.ylabel(zh_name)
        plt.title(f"{zh_name} 对比")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # 用 key 或汉字名来命名文件
        # 把空格和括号去掉，以免路径怪怪的
        safe_name = key.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = os.path.join(log_dir, f"{safe_name}.png")

        plt.savefig(save_path, bbox_inches='tight')
        print(f"已保存图像: {save_path}")

        # 如果你只想保存不弹窗，可以把下一行注释掉
        # plt.show()

        plt.close()


if __name__ == "__main__":
    draw()
