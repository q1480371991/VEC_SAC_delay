# TD3 Agent with FDN diffusion actor
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import Critic
# 引入 FDN 扩散策略
from policy_diffusion import DiffusionPolicyWrapper

# 设备设定
GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx))
else:
    device = torch.device("cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class TD3_FDN_Agent:
    """
    TD3 with diffusion actor:
    - actor: DiffusionPolicyWrapper
    - twin critics, target networks
    - target policy smoothing and delayed policy updates
    """
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=512,
            action_range=1.0,
            replay_buffer=None,
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            denoising_steps=5,
            t_dim=16,
            beta_schedule='vp',
            fdn_noise_scale=1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_range = action_range

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.replay_buffer = replay_buffer

        # diffusion actors
        self.actor = DiffusionPolicyWrapper(state_dim, action_dim, hidden_dim,
                                            denoising_steps=denoising_steps, t_dim=t_dim,
                                            beta_schedule=beta_schedule,
                                            action_range=action_range, device=device).to(device)
        self.actor_target = DiffusionPolicyWrapper(state_dim, action_dim, hidden_dim,
                                                   denoising_steps=denoising_steps, t_dim=t_dim,
                                                   beta_schedule=beta_schedule,
                                                   action_range=action_range, device=device).to(device)
        # 复制参数到 target
        self.actor_target.load_state_dict(self.actor.state_dict())

        # critics
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        # optimizers and loss
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.mse_loss = nn.MSELoss()

        self.total_it = 0
        self.fdn_noise_scale = fdn_noise_scale  # rollout 时扩散噪声尺度

    def get_action(self, state, deterministic=False, explore_noise=0.1, history_action=None):
        """
        使用扩散策略生成动作:
        - deterministic=True: 采用确定性扩散路径（不加噪声），用于评估
        - explore_noise: 在扩散采样中的 noise_scale（而不是直接加高斯到动作），更自然地探索
        """
        state_t = np.asarray(state).flatten()
        return self.actor.get_action(state_t,
                                     history_action=history_action,
                                     noise_scale=self.fdn_noise_scale if not deterministic else 0.0,
                                     deterministic=deterministic)

    def update(self, batch_size=256):
        """
        TD3 更新（twin Q + target smoothing + delayed policy updates）
        - actor 用确定性扩散路径参与训练，保证梯度可达
        """
        if self.replay_buffer is None:
            raise ValueError("Replay buffer is not set for TD3FDNAgent.")
        if len(self.replay_buffer) < batch_size:
            return None, None

        self.total_it += 1

        # sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # target policy smoothing：先用 target 扩散 actor 的确定性输出，再叠加截断噪声
        with torch.no_grad():
            next_action_det, _ = self.actor_target.evaluate(next_state, noise_scale=0.0, deterministic=True)
            noise = torch.randn_like(next_action_det) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action_det + noise).clamp(-self.action_range, self.action_range)

            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        # 当前 Q 估计
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        loss_q1 = self.mse_loss(current_q1, target_q)
        loss_q2 = self.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic2_optimizer.step()

        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            # 确定性扩散路径，保证可微
            actor_action_det, _ = self.actor.evaluate(state, noise_scale=0.0, deterministic=True)
            actor_loss = -self.critic1(state, actor_action_det).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft updates
            self.soft_update_actor()
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

        return (loss_q1.item() + loss_q2.item()) * 0.5, None if actor_loss is None else actor_loss.item()

    def soft_update(self, source_net, target_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def soft_update_actor(self):
        # actor 与 actor_target 参数软更新
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor_fdn.pth'))
        torch.save(self.critic1.state_dict(), os.path.join(path, 'critic1.pth'))
        torch.save(self.critic2.state_dict(), os.path.join(path, 'critic2.pth'))

    def load_model(self, path, map_location='cpu'):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor_fdn.pth'), map_location=map_location))
        self.critic1.load_state_dict(torch.load(os.path.join(path, 'critic1.pth'), map_location=map_location))
        self.critic2.load_state_dict(torch.load(os.path.join(path, 'critic2.pth'), map_location=map_location))
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()