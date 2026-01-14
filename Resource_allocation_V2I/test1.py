import pickle
import os

import numpy as np
from matplotlib import pyplot as plt


def save_results(name, index, E_total, reward, calculate, overload, eta1, load_rate_0, delay ,elapsed_list):
    log_dir = f'../log/DMO_episode400_20260114_000433'
    os.makedirs(log_dir, exist_ok=True)
    data = {
        'Sum_E_total': E_total,
        'Sum_reward': reward,
        'Sum_calculate': calculate,
        'Sum_overload': overload,
        'Sum_eta1': eta1,
        'Sum_load_rate_0': load_rate_0,
        'Sum_delay': delay,
        'elapsed':elapsed_list
    }
    with open(f'{log_dir}/{name}_data_{index}.pkl', 'wb') as f:
        pickle.dump(data, f)

    # 绘图部分 (保持不变)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    metrics = {
        '能量消耗': E_total,
        '奖励': reward,
        '计算量': calculate,
        '过载量': overload,
        '资源浪费率': eta1,
        '卸载率': load_rate_0,
        '时延': delay,
        '耗时':elapsed_list
    }
    x = np.arange(len(E_total))
    for metric_name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(x, values, label='DMO数据', alpha=0.6)  # Label 改一下
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
def modify_reward_sign():
    """
    读取pkl文件，将Sum_reward取负值，然后重新保存
    """
    # 构建文件路径
    log_dir = f'./log/DMO/DMO_episode400_20260114_000433'
    file_path = f'{log_dir}/DMO_data_0.pkl'

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False

    try:
        # 1. 读取数据
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print("修改前数据:", {k: v for k, v in data.items()})

        # 2. 修改Sum_reward为负值
        if 'Sum_reward' in data:
            # 如果Sum_reward是列表，对每个元素取负
            if isinstance(data['Sum_reward'], list):
                data['Sum_reward'] = [-x for x in data['Sum_reward']]
            else:  # 如果是单个数值
                data['Sum_reward'] = -data['Sum_reward']
        else:
            print("警告: 数据中没有找到Sum_reward字段")
            return False

        print("修改后数据:", {k: v for k, v in data.items()})

        # 3. 重新保存数据
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"成功修改并保存: {file_path}")
        return True

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False


# 批量处理多个文件的函数
def batch_modify_rewards(name, n_episode_test, current_time, indices=None):
    """
    批量处理多个文件
    indices: 要处理的文件索引列表，如果为None则处理所有匹配的文件
    """
    log_dir = f'../log/{name}_episode{n_episode_test}_{current_time}'

    if not os.path.exists(log_dir):
        print(f"目录不存在: {log_dir}")
        return False

    if indices is None:
        # 自动查找所有匹配的文件
        pattern = f'{name}_data_'
        files = [f for f in os.listdir(log_dir) if f.startswith(pattern) and f.endswith('.pkl')]
        indices = []
        for f in files:
            try:
                # 提取索引号
                idx_str = f.replace(f'{name}_data_', '').replace('.pkl', '')
                indices.append(int(idx_str))
            except:
                continue

    success_count = 0
    for index in indices:
        if modify_reward_sign(name, index, n_episode_test, current_time):
            success_count += 1

    print(f"处理完成: 成功{success_count}/{len(indices)}个文件")
    return success_count


# 如果你想要修改原始save_results函数，使其直接保存负的reward：
def save_results_modified(name, index, E_total, reward, calculate, overload, eta1, load_rate_0, delay, elapsed_list):
    """
    修改版的save_results，直接保存负的reward
    """
    # 在保存之前将reward取负
    reward_neg = [-x for x in reward] if isinstance(reward, list) else -reward

    # 调用原始函数，但传入修改后的reward
    save_results(name, index, E_total, reward_neg, calculate, overload, eta1, load_rate_0, delay, elapsed_list)


# 使用示例
if __name__ == "__main__":
    # 单个文件修改示例
    modify_reward_sign()
