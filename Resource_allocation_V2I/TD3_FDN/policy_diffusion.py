# policy_diffusion.py
# 扩散策略包装器：提供 sample_action / evaluate 接口，并做动作范围与约束映射

import torch
import numpy as np
from typing import Optional, Tuple
from diffusion_continuous import ContinuousDiffusion
from denoiser_mlp import ContinuousDenoiserMLP


class DiffusionPolicyWrapper(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int,
                 denoising_steps: int = 5, t_dim: int = 16,
                 beta_schedule: str = 'vp',
                 action_range: Optional[float] = 1.0,
                 bounds: Optional[dict] = None,
                 device: Optional[torch.device] = torch.device('cpu')):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.action_range = action_range
        self.bounds = bounds  # 可选：{'power':(min,max), 'cpu':(min,max), 'rsu':(0,1)}

        model = ContinuousDenoiserMLP(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, t_dim=t_dim)
        self.diff = ContinuousDiffusion(state_dim=state_dim, action_dim=action_dim, model=model,
                                        beta_schedule=beta_schedule, denoising_steps=denoising_steps).to(device)

    def map_action(self, x0: torch.Tensor) -> torch.Tensor:
        # 将连续 x0 原型映射到环境期望动作范围
        a = torch.tanh(x0)
        if self.action_range is not None:
            a = self.action_range * a
        return a

    def evaluate(self, state: torch.Tensor, history_action: Optional[np.ndarray] = None,
                 noise_scale: float = 1.0, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作：
        - deterministic=True：采用确定性均值轨迹，不加噪声，用于 actor 训练的可微路径
        - 若提供 history_action，则用其归一到 [-1,1] 后作为 init_x 热启动，再叠加 noise_scale * N(0,I)
        """
        init_x = None
        if history_action is not None:
            ha = torch.as_tensor(history_action, dtype=torch.float32, device=self.device)
            if ha.dim() == 1:
                ha = ha.unsqueeze(0)  # (1, action_dim)
            scale = self.action_range if self.action_range is not None else 1.0
            z = torch.clamp(ha / scale, -0.999, 0.999)
            init_x = z  # 直接作为未压缩空间的初值

        x0 = self.diff.sample(state, init_x=init_x, noise_scale=noise_scale, deterministic=deterministic)
        a = self.map_action(x0)
        return a, x0

    def get_action(self, state_np: np.ndarray, history_action: Optional[np.ndarray] = None,
                   noise_scale: float = 1.0, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.evaluate(state, history_action=history_action, noise_scale=noise_scale, deterministic=deterministic)
        return a.detach().cpu().numpy()[0]

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.log((1 + x) / (1 - x + 1e-8))