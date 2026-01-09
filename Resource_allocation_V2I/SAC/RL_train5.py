'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''

'''
Soft Actor-Critic version 2（软演员-评论家算法版本2）
使用目标Q网络而非V网络：包含2个Q网络、2个目标Q网络、1个策略网络
与版本1相比增加了alpha损失（熵温度参数的损失计算）
参考论文：https://arxiv.org/pdf/1812.05905.pdf
'''

import math
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# from IPython.display import clear_output
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from IPython.display import display
#
# import argparse
# import time

GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx))

else:
    device = torch.device("cpu")
print(device)

class ReplayBuffer:
    """经验回放缓冲区，用于存储智能体与环境交互的经验数据"""
    def __init__(self, capacity):
        """初始化缓冲区，capacity为缓冲区最大容量"""
        self.capacity = capacity
        self.buffer = []# 存储经验的列表
        self.position = 0# 当前存储位置（用于环形缓冲区）

    def push(self, state, action, reward, next_state, done):
        """将一条经验（状态、动作、奖励、下一状态、是否结束）存入缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)# 预留位置
        # 存储经验，环形覆盖旧数据
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # 更新位置

    def sample(self, batch_size):
        """从缓冲区中随机采样batch_size条经验"""
        batch = random.sample(self.buffer, batch_size)
        # 将采样的经验按元素类型堆叠（如所有状态堆叠成一个数组）
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ValueNetwork(nn.Module):
    """价值网络（V网络），用于估计状态的价值（仅定义但未在SAC v2中使用，可能为预留）"""
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        # 定义4层全连接网络
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)# 输出状态价值（标量）
        # 初始化最后一层的权重和偏置（小范围随机值）
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """前向传播：输入状态，输出状态价值"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class SoftQNetwork(nn.Module):
    """软Q网络（Q网络），用于估计状态-动作对的Q值"""
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        """
        Soft Q-Learning:神经网络包括四个线性层，其中前三个层使用 ReLU（Rectified Linear Unit）作为激活函数，最后一层没有激活函数。
        在前向传播过程中，输入的状态和动作在维度 1 上拼接，然后通过神经网络的各个层。
        """
        # 输入维度为状态维度+动作维度
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)# 输出Q值（标量）
        # 初始化最后一层的权重和偏置
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        """前向传播：输入状态和动作，输出Q值"""
        x = torch.cat([state, action], 1)  # 在维度1上拼接状态和动作
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class PolicyNetwork(nn.Module):
    """策略网络，用于生成动作分布（高斯分布），输出动作的均值和标准差"""
    def __init__(self, num_inputs, num_actions, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20,
                 log_std_max=2):
        super(PolicyNetwork, self).__init__()
        """
        策略网络：神经网络包括四个隐藏层，分别是线性层，均值和日志标准差通过线性层计算。
        这个模型的输出通常是一个均值向量和一个对数标准差向量，它们用于参数化动作分布。
        在这里，均值和对数标准差的权重和偏置在初始化时使用均匀分布进行初始化。
        """

        self.log_std_min = log_std_min# 对数标准差的最小值（防止标准差过小）
        self.log_std_max = log_std_max# 对数标准差的最大值（防止标准差过大）
        # 四层全连接网络提取特征
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        # 输出动作均值
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        # 输出动作对数标准差
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range# 动作范围（用于缩放输出动作）
        self.num_actions = num_actions# 动作维度

    def forward(self, state):
        """前向传播：输入状态，输出动作的均值和对数标准差"""
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        mean = (self.mean_linear(x))# 动作均值
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)# 动作对数标准差
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)# 限制对数标准差范围

        return mean, log_std  # 均值、对数标准差

    def evaluate(self, state, epsilon=1e-6):
        '''
        在 update 中，策略损失为 E[α log π(a|s) − min(Q1, Q2)(s, a)]，这里的 log π(a|s) 就由 evaluate 返回

        生成基于当前策略的采样动作，并计算动作的对数概率（用于训练时的梯度计算）
        采用重参数化技巧（reparameterization trick）减少方差
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()  # 标准差（指数化对数标准差）

        normal = Normal(0, 1) # 标准正态分布
        z = normal.sample(mean.shape) # 采样噪声
        # 重参数化：动作 = tanh(均值 + 标准差*噪声) * 动作范围（将动作限制在[-action_range, action_range]）
        action_0 = torch.tanh(mean + std * z.to(device))  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.

        # 计算动作的对数概率（考虑tanh压缩和动作范围缩放的修正）
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        # 对动作维度求和，得到标量对数概率
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, deterministic):
        """获取动作：根据当前状态生成动作（用于与环境交互）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device) # 转换状态为张量并添加批次维度
        state = state.reshape(-1, state.size()[1])  #不管进来的state是什么维度，这一步将state第二维调整为8 调整状态维度（确保第二维为状态维度）
        mean, log_std = self.forward(state)
        std = log_std.exp()# 标准差（指数化对数标准差）

        normal = Normal(0, 1)#高斯分布
        z = normal.sample(mean.shape).to(device)
        action = self.action_range * torch.tanh(mean + std * z)# 随机采样动作
        # 如果是确定性模式，直接使用均值的tanh作为动作（无探索噪声）
        action = self.action_range * torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else \
        action.detach().cpu().numpy()[0]
        return action

    def sample_action(self, ):
        """随机采样一个动作（用于初始化等场景）"""
        a = torch.FloatTensor(self.num_actions).uniform_(-1, 1)# 在[-1,1]范围内均匀采样
        return self.action_range * a.numpy() # 缩放到动作范围


class SAC_Trainer():
    """SAC算法训练器，用于协调各个网络的训练和更新"""
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, action_range):
        """初始化训练器：创建网络、优化器，并初始化目标网络"""
        self.replay_buffer = replay_buffer# 经验回放缓冲区

        # 创建两个Q网络和两个目标Q网络
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        # 创建策略网络
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        # 熵温度参数alpha的对数（可学习参数）
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        # 初始化目标Q网络的参数（与Q网络相同）
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)
        # 定义Q网络的损失函数（均方误差损失）
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        # 学习率设置
        soft_q_lr = 3e-4# Q网络学习率
        policy_lr = 3e-4# 策略网络学习率
        alpha_lr = 3e-4# alpha参数学习率
        # 创建优化器
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        """
                更新网络参数：从缓冲区采样数据，依次更新Q网络、策略网络和alpha参数
                batch_size：采样批次大小
                reward_scale：奖励缩放系数
                auto_entropy：是否自动调整alpha
                target_entropy：目标熵（用于alpha更新）
                gamma：折扣因子
                soft_tau：目标网络软更新系数
        """
        # 从缓冲区采样数据
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
        # 转换数据为张量并移动到指定设备
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        # 奖励添加批次维度，并移动到设备
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        # 结束标志转换为浮点型并添加批次维度
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        # 计算当前Q网络对采样动作的Q值估计
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        # 从当前策略网络采样动作，并计算对数概率
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        # 为下一状态采样动作，用于计算目标Q值
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        # 标准化奖励（减去均值，除以标准差），减少奖励尺度对训练的影响
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        # 更新alpha（熵温度参数）
        if auto_entropy is True:
            # alpha损失：使策略的熵接近目标熵
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad() # 清空梯度
            alpha_loss.backward()# 反向传播计算梯度
            self.alpha_optimizer.step() # 更新alpha参数
            self.alpha = self.log_alpha.exp()# 计算alpha（指数化对数）
        else:
            self.alpha = 1. # 固定alpha为1
            alpha_loss = 0

        # 计算目标Q值（使用目标Q网络和下一状态的动作）
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        # 目标Q值 = 奖励 + (1 - 结束标志) * 折扣因子 * 目标Q值（下一状态）
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        # 更新Q网络（最小化预测Q值与目标Q值的均方误差）
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                               target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        # 更新第一个Q网络
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        # 更新第二个Q网络
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # 更新策略网络（最大化Q值减去alpha*对数概率，等价于最小化负的目标）
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()#log_prob 是负的

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # 软更新目标Q网络（缓慢跟踪Q网络的参数）
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        path_q1 = os.path.join(path, 'q1')
        path_q2 = os.path.join(path, 'q2')
        path_policy = os.path.join(path, 'policy')
        torch.save(self.soft_q_net1.state_dict(), path_q1)
        torch.save(self.soft_q_net2.state_dict(), path_q2)
        torch.save(self.policy_net.state_dict(), path_policy)

    def load_model(self, path):
        path_q1 = os.path.join(path, 'q1.pth')
        path_q2 = os.path.join(path, 'q2.pth')
        path_policy = os.path.join(path, 'policy.pth')
        self.soft_q_net1.load_state_dict(torch.load(path_q1, map_location=torch.device('cpu')))
        self.soft_q_net2.load_state_dict(torch.load(path_q2, map_location=torch.device('cpu')))
        self.policy_net.load_state_dict(torch.load(path_policy, map_location=torch.device('cpu')))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
