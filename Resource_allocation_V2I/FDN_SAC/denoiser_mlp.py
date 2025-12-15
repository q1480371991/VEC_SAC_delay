# denoiser_mlp.py
# 连续扩散 Actor 的去噪网络：输入 x_t + t_embed + state，输出噪声预测 ε_hat（action_dim维）

import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers_continuous import SinusoidalPosEmb

class ContinuousDenoiserMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, t_dim: int = 16):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(t_dim)
        self.fc1 = nn.Linear(state_dim + action_dim + t_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # x_t: (batch, action_dim)
        # t:   (batch,)
        # state: (batch, state_dim or any, 会被展平到 2D)
        t_embed = self.time_emb(t)
        state = state.reshape(state.size(0), -1)
        h = torch.cat([x_t , state , t_embed], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        eps_hat = self.fc3(h)  # 线性输出，表示噪声预测
        return eps_hat