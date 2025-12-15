# helpers_continuous.py
# 连续扩散 Actor 的公共工具：时间编码、beta 调度与张量抽取

import math
import numpy as np
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch,) 的整数时间步
        device = x.device
        half = self.dim // 2
        emb_freq = math.log(10000) / (max(half - 1, 1))
        freqs = torch.exp(torch.arange(half, device=device) * -emb_freq)
        x = x.float()
        args = x[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape):
    # 从 1D 系数数组 a 中按时间步 t 采样对应系数，并 reshape 到 x_shape 的批形状
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)

def cosine_beta_schedule(timesteps: int, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def vp_beta_schedule(timesteps: int, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / timesteps - 0.5 * (b_max - b_min) * (2 * t - 1) / timesteps ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)
