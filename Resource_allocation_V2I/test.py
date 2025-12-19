from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

def test():
    # 示例：4个动作的原始分数
    action_logits = torch.tensor([[3.0, 1.0, 0.2, 2.5]])
    print("原始分数 (logits):", action_logits)

    # softmax转换
    action_probs = F.softmax(action_logits, dim=-1)
    print("概率分布:", action_probs)
    print("概率总和:", action_probs.sum().item())  # 应该是1.0
def visualize_action_sampling():
    # 定义一个动作的分布
    mean = 0.0  # 均值：最可能的值
    std = 0.5  # 标准差：探索程度

    # 生成概率密度函数曲线
    x = np.linspace(-2, 2, 100)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label=f'高斯分布(μ={mean}, σ={std})')
    plt.axvline(mean, color='r', linestyle='--', label='均值')

    # 采样几个点
    samples = np.random.normal(mean, std, 5)
    plt.scatter(samples, np.zeros_like(samples), color='green', s=100,
                label='采样点', zorder=5)

    plt.title("动作采样过程可视化")
    plt.xlabel("动作值")
    plt.ylabel("概率密度")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def explain_mean_log_std_shape():
    print("\n=== mean和log_std的形状分析 ===\n")

    # 假设有2个动作的机器人控制任务
    num_actions = 2
    batch_size = 3  # 一次处理3个状态

    # 神经网络输出的mean和log_std形状
    mean = torch.randn(batch_size, num_actions)  # 形状: [3, 2]
    log_std = torch.randn(batch_size, num_actions)  # 形状: [3, 2]

    print(f"mean形状: {mean.shape}")  # [batch_size, num_actions]
    print(f"log_std形状: {log_std.shape}")  # [batch_size, num_actions]
    print(f"\nmean值:\n{mean}")
    print(f"\nlog_std值:\n{log_std}")

    # 解释：每个状态对应每个动作都有一个均值和标准差
    print(f"\n含义解释:")
    print(f"• 批量大小: {batch_size}个状态")
    print(f"• 每个状态有{num_actions}个动作维度")
    print(f"• 每个动作维度都有自己的均值和标准差")

    # 具体例子：机器人控制
    actions = ["关节1角度", "关节2角度"]
    states = ["状态A", "状态B", "状态C"]

    print(f"\n具体含义:")
    for i, state in enumerate(states):
        print(f"{state}:")
        for j, action in enumerate(actions):
            print(f"  {action}: 均值={mean[i, j]:.3f}, 对数标准差={log_std[i, j]:.3f}")


def visualize_sampling_process():
    """可视化高斯分布采样过程"""

    mean = 2.0
    std = 0.5

    # 生成标准正态分布样本
    z_samples = torch.randn(10000)  # z ∼ N(0, 1)

    # 转换到目标分布
    x_samples = mean + std * z_samples  # x ∼ N(2.0, 0.5²)

    plt.figure(figsize=(12, 4))

    # 子图1: 标准正态分布
    plt.subplot(1, 3, 1)
    plt.hist(z_samples.numpy(), bins=50, alpha=0.7, density=True)
    x_range = np.linspace(-4, 4, 100)
    plt.plot(x_range, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x_range ** 2), 'r-')
    plt.title('标准正态分布 z ∼ N(0, 1)')
    plt.xlabel('z值')
    plt.ylabel('概率密度')

    # 子图2: 转换过程
    plt.subplot(1, 3, 2)
    # 显示转换关系
    sample_indices = np.random.choice(len(z_samples), 100, replace=False)
    plt.scatter(z_samples[sample_indices], x_samples[sample_indices], alpha=0.6)
    plt.plot([-3, 3], [mean - 3 * std, mean + 3 * std], 'r--', label='x = μ + σz')
    plt.xlabel('标准正态变量 z')
    plt.ylabel('目标变量 x')
    plt.title('线性变换: x = μ + σz')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 目标分布
    plt.subplot(1, 3, 3)
    plt.hist(x_samples.numpy(), bins=50, alpha=0.7, density=True)
    x_range_target = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    plt.plot(x_range_target, 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_range_target - mean) / std) ** 2), 'r-')
    plt.title(f'目标分布 x ∼ N({mean}, {std}²)')
    plt.xlabel('x值')
    plt.ylabel('概率密度')

    plt.tight_layout()
    plt.show()


# visualize_sampling_process()

if __name__ == "__main__":
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # 访问第一行
    first_row = t[0]
    print(f"\n第一行: {first_row}")  # 输出: tensor([1, 2, 3])

    # 访问所有行和第二列
    second_col = t[:, 1]
    print(f"第二列: {second_col}")  # 输出: tensor([2, 5, 8])

    # 访问右下角的 2x2 子矩阵
    sub_matrix = t[1:, 1:]
    print(f"右下角子矩阵:\n {sub_matrix}")
