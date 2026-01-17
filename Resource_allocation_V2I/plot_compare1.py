import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tabulate import tabulate


def draw():
    # 创建保存目录
    log_dir = 'compare_plot_log'
    os.makedirs(log_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    runs = {
        "GRU_MATD3": "./log/MATD3/MATD3_numv5_test_episode400_20260114_193758/MATD3_test_data_0.pkl",
        "SAC": "./log/origin/SAC_numv5_0_episode400_delay4000/SAC_data_0.pkl",
        "SAC_FDN": "./log/FDNSAC_用历史动作初始化_熵正则/FDN_SAC_episode400_20251212_1310_熵正则A1e2/FDN_SAC_data_0.pkl",
        "Rand": "./log/Rand/Rand_numv5_episode400_20260114_134635/Rand_data_0.pkl",
        "SA": "./log/SA/SA_numv5_episode400_20260114_153221/SA_data_0.pkl",
    }

    metric_keys = {
        "奖励 (reward)": "Sum_reward",
        "时延 (delay)": "Sum_delay",
        "能耗 (E_total)": "Sum_E_total",
        "卸载率 (load_rate_0)": "Sum_load_rate_0",
        "资源浪费率 (eta1)": "Sum_eta1"
    }

    # ======== 读取所有 pkl 数据 ========
    results = {}
    min_length = float('inf')

    # 第一遍：读取数据并找到最小长度
    for name, path in runs.items():
        if not os.path.exists(path):
            print(f"[警告] 文件不存在: {path}")
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        # 找到第一个数组键的长度作为参考
        for key, value in data.items():
            if isinstance(value, (list, np.ndarray)):
                min_length = min(min_length, len(value))
                break

    print(f"\n{'=' * 60}")
    print(f"统一数据长度为: {min_length} episodes")
    print(f"{'=' * 60}\n")

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

    # ======== 打印最后一 episode 的数据 ========
    print("\n" + "=" * 80)
    print("各算法最后一 episode 的数据统计")
    print("=" * 80)

    # 创建数据表格
    table_data = []
    headers = ["算法名称"] + list(metric_keys.keys()) + ["最后 Episode"]

    for name, data in results.items():
        row = [name]
        last_episode = None

        for zh_name, key in metric_keys.items():
            if key in data and isinstance(data[key], (list, np.ndarray)):
                if len(data[key]) > 0:
                    last_value = data[key][-1]  # 最后一 episode 的数据
                    row.append(f"{last_value:.4f}" if isinstance(last_value, (int, float)) else str(last_value))
                    last_episode = len(data[key])  # 记录 episode 数
                else:
                    row.append("N/A")
            else:
                row.append("N/A")

        row.append(str(last_episode) if last_episode else "N/A")
        table_data.append(row)

    # 使用 tabulate 美化输出
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="right"))

    # # 额外输出每个指标的最佳算法
    # print("\n" + "=" * 80)
    # print("各指标最优算法（最后一 episode）")
    # print("=" * 80)
    #
    # for zh_name, key in metric_keys.items():
    #     best_algo = None
    #     best_value = None
    #     is_better = None
    #
    #     for name, data in results.items():
    #         if key in data and isinstance(data[key], (list, np.ndarray)) and len(data[key]) > 0:
    #             value = data[key][-1]
    #
    #             if best_algo is None:
    #                 best_algo = name
    #                 best_value = value
    #                 continue
    #
    #             # 判断指标优化方向（越大越好还是越小越好）
    #             if "reward" in key.lower() or "load_rate" in key.lower():
    #                 # 奖励和卸载率越大越好
    #                 is_better = "↑"  # 上升趋势表示更好
    #                 if value > best_value:
    #                     best_algo = name
    #                     best_value = value
    #             else:
    #                 # 时延、能耗、资源浪费率越小越好
    #                 is_better = "↓"  # 下降趋势表示更好
    #                 if value < best_value:
    #                     best_algo = name
    #                     best_value = value
    #
    #     if best_algo:
    #         trend_symbol = is_better if is_better else " "
    #         print(f"{zh_name:<15} 最优算法: {best_algo:<10} 值: {best_value:.4f} {trend_symbol}")

    # ======== 绘图并保存 ========
    print(f"\n开始生成对比图...")
    for zh_name, key in metric_keys.items():
        plt.figure(figsize=(10, 6))

        for name, data in results.items():
            if key not in data:
                continue

            y = data[key]
            x = range(len(y))
            plt.plot(x, y, label=name, linewidth=2)

            # 在图中标记最后一 episode 的值
            if len(y) > 0:
                plt.annotate(f'{y[-1]:.4f}',
                             xy=(len(y) - 1, y[-1]),
                             xytext=(len(y) - 5, y[-1] * 1.05),
                             arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

        plt.xlabel("训练轮次 / Episode", fontsize=12)
        plt.ylabel(zh_name, fontsize=12)
        plt.title(f"{zh_name} 对比", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        safe_name = key.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = os.path.join(log_dir, f"{safe_name}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  已保存: {save_path}")
        plt.close()

    # 保存数据到CSV以便进一步分析
    csv_path = os.path.join(log_dir, "last_episode_data.csv")
    df_data = []
    for name, data in results.items():
        for zh_name, key in metric_keys.items():
            if key in data and isinstance(data[key], (list, np.ndarray)) and len(data[key]) > 0:
                df_data.append({
                    "算法": name,
                    "指标": zh_name,
                    "指标键名": key,
                    "最后一episode值": data[key][-1],
                    "均值": np.mean(data[key]) if len(data[key]) > 0 else None,
                    "标准差": np.std(data[key]) if len(data[key]) > 0 else None
                })

    if df_data:
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n详细数据已保存到: {csv_path}")

    print(f"\n{'=' * 60}")
    print("数据统计和图表生成完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    draw()