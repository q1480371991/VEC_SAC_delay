import numpy as np
import copy


class DMOSolver:
    def __init__(self, n_veh, n_dim, action_bounds, pop_size=20, max_iter=50):
        """
        :param n_veh: 车辆数量
        :param n_dim: 动作维度 (n_veh * 3)
        :param action_bounds: 动作的上下界 [(min, max), (min, max), ...]
        :param pop_size: 种群大小 (论文中的 M)
        :param max_iter: 每次决策的最大迭代次数 (T)
        """
        self.n_veh = n_veh
        self.n_dim = n_dim
        self.bounds = np.array(action_bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter

        # DMO 参数
        self.n_babysitter = 3  # 保姆数量
        self.alpha_group_size = self.pop_size - self.n_babysitter  # Alpha组大小 (L)
        self.peep = 2  # 叫声参数

    def _initialize_population(self):
        # 随机初始化种群 X: [pop_size, n_dim]
        # 根据 bounds 生成
        min_b = self.bounds[:, 0]
        max_b = self.bounds[:, 1]
        X = np.random.uniform(low=min_b, high=max_b, size=(self.pop_size, self.n_dim))
        return X

    def _evaluate(self, X, env_snapshot, state_snapshot):
        """
        计算适应度。注意：我们需要一种方法在不推演环境时间的情况下计算 Reward。
        这里我们假设利用 env 的公式进行计算，但不改变环境内部状态。
        """
        fitness = np.zeros(self.pop_size)

        # 对种群中每个个体计算 Reward
        for i in range(self.pop_size):
            action_flat = X[i]
            # 还原 action 形状为 [n_veh, 3]
            action_matrix = np.zeros((self.n_veh, 3))
            for v in range(self.n_veh):
                idx = v * 3
                action_matrix[v, 0] = action_flat[idx + 0]  # Power
                action_matrix[v, 1] = action_flat[idx + 1]  # Frequency
                action_matrix[v, 2] = action_flat[idx + 2]  # Offload Rate

            # 调用环境的计算函数 (需要 env 提供不更新状态的 reward 计算接口)
            # 这里我们模拟 train_td3.py 中的逻辑，但只计算不 step
            # 注意：这里需要根据你的 Environment3.py 逻辑稍作适配，
            # 假设我们传入的是当前的 env 对象，我们只调用计算部分

            # 1. 计算计算量
            comp_n_list_true, comp_n_list = env_snapshot.true_calculate_num(action_matrix)
            comp_n_list_RSU = env_snapshot.calculate_num_RSU()

            # 注意：Environment3 的 update_buffer 会改变内部状态，DMO 搜索时不应该改变
            # 所以我们需要使用 env 的深拷贝或者修改 env 增加 "simulate_reward" 方法
            # 为简单起见，这里假设 env_snapshot 是一个已经 step 过的环境副本或者我们只计算瞬时 Reward

            # 构建 offload_num
            offload_num = []
            for k in range(self.n_veh):
                # 注意：这里逻辑要和 train_td3 保持一致
                offload_num_i = int(action_matrix[k, 2] * comp_n_list_RSU)
                offload_num.append(offload_num_i)

            # 模拟信道 (使用当前时隙的信道，不生成新的)
            # train_td3 中是在循环里 env.overall_channel(time_slots.now())
            # 这里直接复用 env_snapshot 里的状态

            # 计算 Reward
            # 注意：trans_energy_RSU 需要 h_i_dB，这个存储在 env 中
            # 我们需要确保 h_i_dB 是当前状态的
            h_i_dB = env_snapshot.V2I_channels_abs
            trans_energy = env_snapshot.trans_energy_RSU(action_matrix, h_i_dB)

            # 计算核心 Reward
            # 为了防止 DMO 搜索时修改了 env.ReplayB_v，我们需要临时备份
            original_buffer = copy.deepcopy(env_snapshot.ReplayB_v)

            try:
                # 调用 RSU_reward1 获取 reward_tot
                _, reward_tot, _, _, _ = env_snapshot.RSU_reward1(
                    action_matrix, comp_n_list_true, trans_energy, offload_num
                )
                fitness[i] = reward_tot  # 我们要最大化 Reward
            finally:
                # 恢复环境 buffer，防止 DMO 迭代影响真实环境
                env_snapshot.ReplayB_v = original_buffer

        return fitness

    def solve(self, env, state):
        """
        执行 DMO 主循环
        :param env: 当前环境对象 (用于计算 Reward)
        :param state: 当前状态 (在本实现中主要依靠 env 内部状态)
        :return: 最佳动作 (flattened)
        """
        # 1. 初始化
        X = self._initialize_population()
        fitness = self._evaluate(X, env, state)

        # 记录全局最优
        best_idx = np.argmax(fitness)
        best_X = X[best_idx].copy()
        best_fit = fitness[best_idx]

        # 2. 迭代优化
        for t in range(self.max_iter):
            # --- Alpha Group (觅食) ---
            # 计算雌性首领 (Alpha Female)
            # DMO 概率公式 P = fit_i / sum(fit)
            # 因为 Reward 可能为负，需要处理一下变成正的概率
            fit_shifted = fitness[:self.alpha_group_size] - np.min(fitness) + 1e-6
            prob = fit_shifted / np.sum(fit_shifted)
            alpha_female_idx = np.random.choice(range(self.alpha_group_size), p=prob)
            alpha_female = X[alpha_female_idx]

            new_X = X.copy()

            for i in range(self.alpha_group_size):
                # 论文公式 (10) 模拟觅食
                phi = np.random.uniform(-1, 1)
                new_X[i] = X[i] + phi * (alpha_female - X[i])

            # 边界修正 & 评估 Alpha 组
            new_X = np.clip(new_X, self.bounds[:, 0], self.bounds[:, 1])
            new_fitness = self._evaluate(new_X, env, state)

            # 贪婪更新
            for i in range(self.alpha_group_size):
                if new_fitness[i] > fitness[i]:
                    fitness[i] = new_fitness[i]
                    X[i] = new_X[i]

            # --- Scout Group (侦查) ---
            # 论文公式 (13)
            # 这里简化处理：Alpha 组同时也做侦查或者部分做侦查
            # 在原始 DMO 中，Alpha 组行为后，会计算睡眠丘(Sleeping mound)
            # 这里为了适配实时控制，我们采用简化版：
            # 如果长时间未更新（这里简化为每轮），执行随机扰动

            # --- Babysitter Exchange (保姆交换) ---
            # 重置最差的解
            sorted_indices = np.argsort(fitness)
            worst_idx = sorted_indices[0]  # 最小 fitness

            # 生成新解替换最差解
            X[worst_idx] = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.n_dim)
            # 重新评估被替换的个体 (这里简单起见下轮评估，或者现在评估)

            # 更新全局最优
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fit:
                best_fit = fitness[current_best_idx]
                best_X = X[current_best_idx].copy()

        return best_X