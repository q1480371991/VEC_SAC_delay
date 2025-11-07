import os
import pickle
import numpy as np
import pandas as pd
import openpyxl


def pkl_to_excel(pkl_file_path, excel_save_dir=None):
    """
    将pkl文件转换为Excel文件（适配之前保存的训练数据格式）
    :param pkl_file_path: pkl文件的完整路径（如 'log/SAC_1/SAC_data_1.pkl'）
    :param excel_save_dir: Excel保存目录，默认与pkl文件同目录
    """
    # 1. 检查文件是否存在
    if not os.path.exists(pkl_file_path):
        print(f"错误：文件 {pkl_file_path} 不存在！")
        return

    # 2. 读取pkl文件数据
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"读取pkl文件失败：{e}")
        return

    # 3. 验证数据格式是否匹配（检查是否包含所有预期的键）
    expected_keys = {
        'Sum_E_total', 'Sum_reward', 'Sum_calculate',
        'Sum_overload', 'Sum_eta1', 'Sum_load_rate_0', 'Sum_delay'
    }
    if not expected_keys.issubset(data.keys()):
        missing_keys = expected_keys - set(data.keys())
        print(f"数据格式不匹配，缺少键：{missing_keys}")
        return

    # 4. 整理数据（统一长度，用NaN填充短列表）
    # 获取所有列表的最大长度（确保Excel列对齐）
    max_len = max(
        len(data['Sum_E_total']),
        len(data['Sum_reward']),
        len(data['Sum_calculate']),
        len(data['Sum_overload']),
        len(data['Sum_eta1']),
        len(data['Sum_load_rate_0']),
        len(data['Sum_delay'])
    )

    # 构造Excel数据字典（包含"训练轮次"列）
    excel_data = {
        '训练轮次': np.arange(1, max_len + 1),  # 轮次从1开始计数
        '能量消耗': np.pad(data['Sum_E_total'], (0, max_len - len(data['Sum_E_total'])), constant_values=np.nan),
        '奖励': np.pad(data['Sum_reward'], (0, max_len - len(data['Sum_reward'])), constant_values=np.nan),
        '计算量': np.pad(data['Sum_calculate'], (0, max_len - len(data['Sum_calculate'])), constant_values=np.nan),
        '过载量': np.pad(data['Sum_overload'], (0, max_len - len(data['Sum_overload'])), constant_values=np.nan),
        '过载率': np.pad(data['Sum_eta1'], (0, max_len - len(data['Sum_eta1'])), constant_values=np.nan),
        '负载率': np.pad(data['Sum_load_rate_0'], (0, max_len - len(data['Sum_load_rate_0'])), constant_values=np.nan),
        '时延': np.pad(data['Sum_delay'], (0, max_len - len(data['Sum_delay'])), constant_values=np.nan)
    }

    # 5. 确定Excel保存路径
    if excel_save_dir is None:
        # 默认与pkl文件同目录
        excel_save_dir = os.path.dirname(pkl_file_path)
    os.makedirs(excel_save_dir, exist_ok=True)  # 确保目录存在

    # 生成Excel文件名（与pkl文件同名，替换后缀）
    pkl_file_name = os.path.basename(pkl_file_path)
    excel_file_name = pkl_file_name.replace('.pkl', '.xlsx')
    excel_file_path = os.path.join(excel_save_dir, excel_file_name)

    # 6. 保存为Excel
    try:
        df = pd.DataFrame(excel_data)
        df.to_excel(excel_file_path, index=False)  # 不保存行索引
        print(f"Excel文件已保存至：{excel_file_path}")
    except Exception as e:
        print(f"保存Excel失败：{e}")


# ------------------------------
# 示例用法
# ------------------------------
if __name__ == "__main__":
    # 替换为你的pkl文件路径
    pkl_path = "log/SAC_0_episode300_03/SAC_data_0.pkl"  # 例如：log目录下的SAC_1文件夹中的pkl文件
    pkl_to_excel(pkl_path)

    # 如需批量转换目录下所有pkl文件，可使用以下代码：
    # log_root = "log"  # 根目录
    # for root, dirs, files in os.walk(log_root):
    #     for file in files:
    #         if file.endswith(".pkl"):
    #             pkl_file = os.path.join(root, file)
    #             pkl_to_excel(pkl_file)