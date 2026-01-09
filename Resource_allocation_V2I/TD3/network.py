# Actor 和 Critic 网络定义（PyTorch）
# 风格尽量与 RL_train5.py 中的网络一致（4 层全连接，ReLU，最后一层小范围初始化）
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    确定性 Actor 网络：输入 state -> 输出 action（tanh 缩放到 action_range）
    结构参考 RL_train5.PolicyNetwork 的隐藏层风格（但不输出 mean/log_std）
    """
    def __init__(self, state_dim, action_dim, hidden_size, action_range=1.0, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.action_range = action_range

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        mean = self.mean_linear(x)
        action = torch.tanh(mean) * self.action_range
        return action


class Critic(nn.Module):
    """
    单个 Q 网络（用于构成双 Q）。输入 (state, action) -> 输出 scalar Q
    结构和 RL_train5.SoftQNetwork 保持一致风格
    """
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        q = self.linear4(x)
        return q