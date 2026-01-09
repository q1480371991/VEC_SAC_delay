# TD3 Agent 实现（含 ReplayBuffer、update、save/load）
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import Actor, Critic

# 设备设定（与仓库中 RL_train5.py 一致风格）
GPU = False
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx))
else:
    device = torch.device("cpu")

class ReplayBuffer:
    """与 RL_train5.py 功能兼容的简单回放缓冲区"""
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


class TD3Agent:
    """
    TD3 训练器
    - actor, critic1, critic2
    - target networks
    - update() 实现 twin critics, target policy smoothing, delayed policy updates
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

        # replay buffer（外部传入或内部 None）
        self.replay_buffer = replay_buffer

        # networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(device)

        # copy params to targets
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        # optimizers and loss
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.mse_loss = nn.MSELoss()

        # internal update counter
        self.total_it = 0

    def get_action(self, state, deterministic=False, explore_noise=0.1):
        """
        state: numpy 1D array (flattened)
        deterministic: True -> actor deterministic output (用于 eval)
        explore_noise: 在动作上加入 N(0, explore_noise)（用于采样/探索）
        返回 numpy 数组，长度为 action_dim，值在 [-action_range, action_range]
        """
        # convert to tensor
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        # 保证形状与 RL_train5.PolicyNetwork.get_action 一致的 reshape 防护
        if len(state_t.shape) == 2:
            pass
        else:
            state_t = state_t.reshape(-1, self.state_dim)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        if deterministic:
            return action
        # add gaussian exploration noise
        if explore_noise is not None and explore_noise > 0.0:
            action = action + np.random.normal(0, explore_noise, size=action.shape)
        # clip to bounds
        action = np.clip(action, -self.action_range, self.action_range)
        return action

    def update(self, batch_size=256):
        """
        从 replay buffer 采样并执行一次 TD3 更新（包含可能的延迟 actor 更新）
        返回 critic_loss_mean, actor_loss (None if not updated this step)
        """
        if self.replay_buffer is None:
            raise ValueError("Replay buffer is not set for TD3Agent.")
        if len(self.replay_buffer) < batch_size:
            return None, None

        self.total_it += 1

        # sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # 转换为 tensor
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # target policy smoothing: 在 target actor 输出上加入截断噪声
        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip).to(device)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.action_range, self.action_range)

            # target Q 值：使用 twin targets 的 min
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        # 当前 Q 估计
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # critic loss
        loss_q1 = self.mse_loss(current_q1, target_q)
        loss_q2 = self.mse_loss(current_q2, target_q)

        # update critics
        self.critic1_optimizer.zero_grad()
        loss_q1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss_q2.backward()
        self.critic2_optimizer.step()

        actor_loss = None
        # delayed policy updates
        if self.total_it % self.policy_delay == 0:
            # actor loss: maximize Q1(s, actor(s)) <=> minimize -Q1
            actor_action = self.actor(state)
            actor_loss = -self.critic1(state, actor_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update targets
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)

        return (loss_q1.item() + loss_q2.item()) * 0.5, None if actor_loss is None else actor_loss.item()

    def soft_update(self, source_net, target_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic1.state_dict(), os.path.join(path, 'critic1.pth'))
        torch.save(self.critic2.state_dict(), os.path.join(path, 'critic2.pth'))

    def load_model(self, path, map_location='cpu'):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth'), map_location=map_location))
        self.critic1.load_state_dict(torch.load(os.path.join(path, 'critic1.pth'), map_location=map_location))
        self.critic2.load_state_dict(torch.load(os.path.join(path, 'critic2.pth'), map_location=map_location))
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()