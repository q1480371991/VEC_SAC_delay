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
        # 方案A：保持与原 SAC 一致，tanh 到 [-1,1] 再乘 action_range
        a = torch.tanh(x0)
        if self.action_range is not None:
            a = self.action_range * a

        # 方案B（可选）：直接映射到物理边界（需要你提供 bounds），示意如下：
        # if self.bounds is not None:
        #     a0 = torch.sigmoid(x0[:, 0]) * (self.bounds['power'][1] - self.bounds['power'][0]) + self.bounds['power'][0]
        #     a1 = torch.sigmoid(x0[:, 1]) * (self.bounds['cpu'][1] - self.bounds['cpu'][0]) + self.bounds['cpu'][0]
        #     a2 = torch.sigmoid(x0[:, 2])  # [0,1]
        #     a = torch.stack([a0, a1, a2], dim=1)

        return a

    def evaluate(self, state: torch.Tensor, history_action: Optional[np.ndarray] = None,
                 noise_scale: float = 1.0) -> Tuple[torch.Tensor, None]:
        """
        采样动作：
        - 若提供 history_action，则用其反映射作为 init_x 热启动，并叠加 noise_scale * N(0,I)
        - 返回动作 a（已做 tanh+action_range 映射），log_prob 占位 None
        """
        init_x = None
        if history_action is not None:
            ha = torch.as_tensor(history_action, dtype=torch.float32, device=self.device)
            if ha.dim() == 1:
                ha = ha.unsqueeze(0)  # (1, action_dim)
            # 归一到 [-1,1]
            scale = self.action_range if self.action_range is not None else 1.0
            z = torch.clamp(ha / scale, -0.999, 0.999)

            init_x=z

            # 反映射到未压缩空间   不需要反映射，因为传进来的action还没有映射到 [-action_range, action_range]，只是裁剪到[-0.999, 0.999]（避免极端值）
            # init_x = self.atanh(z)

        x0 = self.diff.sample(state, init_x=init_x, noise_scale=noise_scale)
        a = self.map_action(x0)
        return a, x0  #返回原始 x0（未 tanh/未 action_range）以便构造熵代理
    # def evaluate(self, state: torch.Tensor) -> Tuple[torch.Tensor, None]:
    #     # 训练时调用：返回动作与占位的 log_prob(None)
    #     self.eval()  # 采样阶段不启用dropout等
    #     x0 = self.diff.sample(state)
    #     a = self.map_action(x0)
    #     return a, None  # SAC的log_prob暂不使用

    def get_action(self, state_np: np.ndarray, history_action: Optional[np.ndarray] = None,
                   noise_scale: float = 1.0, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.evaluate(state, history_action=history_action, noise_scale=noise_scale)
        return a.detach().cpu().numpy()[0]

    @staticmethod
    def atanh(x: torch.Tensor) -> torch.Tensor:
        # 数值稳定的 atanh，输入需在 (-1,1) 内
        return 0.5 * torch.log((1 + x) / (1 - x + 1e-8))