# diffusion_continuous.py
# 连续扩散采样：不做 softmax，直接生成连续 x0；由策略包装器做动作约束映射

import numpy as np
import torch
import torch.nn as nn
from helpers_continuous import (
    extract_into_tensor,
    linear_beta_schedule,
    cosine_beta_schedule,
    vp_beta_schedule,
)

class ContinuousDiffusion(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, model: nn.Module,
                 beta_schedule: str = 'vp', denoising_steps: int = 5, predict_epsilon: bool = True,
                 clip_denoised: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = model
        self.predict_epsilon = predict_epsilon
        self.clip_denoised = clip_denoised

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(denoising_steps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(denoising_steps)
        else:
            betas = vp_beta_schedule(denoising_steps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(denoising_steps)
        # 注册 buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 预计算系数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        # x0 ≈ sqrt(1/ᾱ_t) * x_t - sqrt(1/ᾱ_t - 1) * ε_hat
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, state: torch.Tensor):
        # 模型预测噪声 eps_hat
        eps_hat = self.model(x_t, t, state)
        x0_hat = self.predict_start_from_noise(x_t, t=t, noise=eps_hat)
        if self.clip_denoised:
            x0_hat = torch.clamp(x0_hat, -1., 1.)
        mean, variance, log_variance = self.q_posterior(x_start=x0_hat, x_t=x_t, t=t)
        return mean, variance, log_variance

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, state: torch.Tensor):
        b, *_, device = *x_t.shape, x_t.device
        mean, _, log_variance = self.p_mean_variance(x_t=x_t, t=t, state=state)
        noise = torch.randn_like(x_t)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * (0.5 * log_variance).exp() * noise

    def sample(self, state: torch.Tensor, init_x: torch.Tensor = None, noise_scale: float = 1.0) -> torch.Tensor:
        """
        从初始 x_T 开始反向扩散：
        - 若提供 init_x（历史动作反映射后的未压缩空间），则 x_T = init_x + noise_scale * N(0, I)
        - 否则从纯噪声开始：x_T = noise_scale * N(0, I)
        """
        device = self.betas.device
        batch_size = state.shape[0]
        if init_x is None:
            x_t = noise_scale * torch.randn(batch_size, self.action_dim, device=device)
        else:
            # 保留噪声，避免历史动作质量差时过度偏置
            x_t = init_x + noise_scale * torch.randn_like(init_x)

        for i in reversed(range(0, self.n_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(x_t, t, state)
        return x_t

    # def sample(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    #     # 从纯噪声开始反向扩散，得到连续 x0
    #     device = self.betas.device
    #     batch_size = state.shape[0]
    #     x_t = torch.randn(batch_size, self.action_dim, device=device)
    #     for i in reversed(range(0, self.n_timesteps)):
    #         t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    #         x_t = self.p_sample(x_t, t, state)
    #     return x_t  # 连续动作原型（未做范围映射）