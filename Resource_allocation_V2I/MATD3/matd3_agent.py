import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import Actor, Critic

def get_device(use_gpu: bool = True, device_idx: int = 0):
    if use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{device_idx}")
    return torch.device("cpu")

class MultiCritic(nn.Module):
    """
    向量 Critic：输入 (state_all, action_all) -> 输出每智能体一个 Q 值，shape=(batch, n_veh)
    """
    def __init__(self, state_dim_total, action_dim_total, hidden_size, n_veh, init_w=3e-3):
        super().__init__()
        self.n_veh = n_veh
        self.l1 = nn.Linear(state_dim_total + action_dim_total, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_veh)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state_all, action_all):
        x = torch.cat([state_all, action_all], dim=1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        q_all = self.out(x)
        return q_all

class MATD3Agent:
    """
    MATD3（中心化训练、去中心化执行）
    - 支持两种 Critic 形态：标量（共享全局 Q）或向量（每智能体一个 Q_i）
    - 支持 Actor 参数共享或独立
    """
    def __init__(
        self,
        n_veh: int,
        obs_dim_per_agent: int = 2,
        act_dim_per_agent: int = 3,
        hidden_dim: int = 512,
        action_range: float = 1.0,
        replay_buffer=None,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        actor_shared: bool = True,
        critic_vector: bool = False,
        use_gpu: bool = True,
        device_idx: int = 0,
    ):
        self.n_veh = n_veh
        self.obs_dim = obs_dim_per_agent
        self.act_dim = act_dim_per_agent
        self.state_dim_total = self.obs_dim * self.n_veh
        self.action_dim_total = self.act_dim * self.n_veh

        self.hidden_dim = hidden_dim
        self.action_range = action_range

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.actor_shared = actor_shared
        self.critic_vector = critic_vector

        self.replay_buffer = replay_buffer

        self.device = get_device(use_gpu=use_gpu, device_idx=device_idx)

        # Actors
        if self.actor_shared:
            self.actor = Actor(self.obs_dim, self.act_dim, self.hidden_dim, self.action_range).to(self.device)
            self.actor_target = Actor(self.obs_dim, self.act_dim, self.hidden_dim, self.action_range).to(self.device)
            self.actors = None
            self.actors_target = None
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        else:
            self.actors = nn.ModuleList(
                [Actor(self.obs_dim, self.act_dim, self.hidden_dim, self.action_range) for _ in range(self.n_veh)]
            ).to(self.device)
            self.actors_target = nn.ModuleList(
                [Actor(self.obs_dim, self.act_dim, self.hidden_dim, self.action_range) for _ in range(self.n_veh)]
            ).to(self.device)
            self.actor = None
            self.actor_target = None
            self.actor_optimizers = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]

        # Critics (twin)
        if self.critic_vector:
            self.critic1 = MultiCritic(self.state_dim_total, self.action_dim_total, self.hidden_dim, self.n_veh).to(self.device)
            self.critic2 = MultiCritic(self.state_dim_total, self.action_dim_total, self.hidden_dim, self.n_veh).to(self.device)
            self.critic1_target = MultiCritic(self.state_dim_total, self.action_dim_total, self.hidden_dim, self.n_veh).to(self.device)
            self.critic2_target = MultiCritic(self.state_dim_total, self.action_dim_total, self.hidden_dim, self.n_veh).to(self.device)
        else:
            self.critic1 = Critic(self.state_dim_total, self.action_dim_total, self.hidden_dim).to(self.device)
            self.critic2 = Critic(self.state_dim_total, self.action_dim_total, self.hidden_dim).to(self.device)
            self.critic1_target = Critic(self.state_dim_total, self.action_dim_total, self.hidden_dim).to(self.device)
            self.critic2_target = Critic(self.state_dim_total, self.action_dim_total, self.hidden_dim).to(self.device)

        # copy params to targets
        self._hard_update(self.critic1_target, self.critic1)
        self._hard_update(self.critic2_target, self.critic2)
        if self.actor_shared:
            self._hard_update(self.actor_target, self.actor)
        else:
            for tgt, src in zip(self.actors_target, self.actors):
                self._hard_update(tgt, src)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.total_it = 0

    def _hard_update(self, target: nn.Module, source: nn.Module):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(s_param.data)

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)

    def _slice_obs(self, state_all: torch.Tensor, i: int) -> torch.Tensor:
        """
        从全局状态中切出第 i 个智能体的局部观测
        state_all: shape (batch, 2*n_veh) 或 (2*n_veh,)
        返回: shape (batch, 2) 或 (2,)
        """
        if state_all.dim() == 1:
            n = self.n_veh
            return torch.stack([state_all[i], state_all[n + i]], dim=0)
        else:
            n = self.n_veh
            return torch.stack([state_all[:, i], state_all[:, n + i]], dim=1)

    def get_actions(self, state_all_np: np.ndarray, deterministic: bool = False, explore_noise: float = 0.0):
        """
        输入：numpy 的全局状态 (2*n_veh,)
        输出：numpy 的 concat 动作 (3*n_veh,)，范围约为 [-1, 1]（train 脚本将映射到物理范围）
        """
        state_all = torch.tensor(state_all_np, dtype=torch.float32, device=self.device).flatten()
        actions = []
        if self.actor_shared:
            for i in range(self.n_veh):
                obs_i = self._slice_obs(state_all, i)
                act_i = self.actor(obs_i.unsqueeze(0)).squeeze(0)
                actions.append(act_i)
        else:
            for i in range(self.n_veh):
                obs_i = self._slice_obs(state_all, i)
                act_i = self.actors[i](obs_i.unsqueeze(0)).squeeze(0)
                actions.append(act_i)
        action_all = torch.cat(actions, dim=0)

        if not deterministic and explore_noise > 0.0:
            noise = torch.randn_like(action_all) * explore_noise
            action_all = action_all + noise
            action_all = torch.clamp(action_all, -1.0, 1.0)

        return action_all.detach().cpu().numpy()

    def _target_actions(self, next_state_all: torch.Tensor):
        """
        生成 target actors 的下一步动作（加入 TD3 policy noise 并裁剪）
        next_state_all: shape (batch, 2*n_veh)
        返回: next_action_all: shape (batch, 3*n_veh)
        """
        batch_size = next_state_all.shape[0]
        actions = []
        if self.actor_shared:
            for i in range(self.n_veh):
                obs_i = self._slice_obs(next_state_all, i)  # (batch, 2)
                act_i = self.actor_target(obs_i)  # (batch, 3)
                actions.append(act_i)
        else:
            for i in range(self.n_veh):
                obs_i = self._slice_obs(next_state_all, i)
                act_i = self.actors_target[i](obs_i)
                actions.append(act_i)
        next_actions_all = torch.cat(actions, dim=1)  # (batch, 3*n_veh)

        # policy noise
        noise = (torch.randn_like(next_actions_all) * self.policy_noise)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_actions_all = next_actions_all + noise
        next_actions_all = torch.clamp(next_actions_all, -1.0, 1.0)

        return next_actions_all

    def update(self, batch_size: int):
        """
        从回放缓冲区采样并更新（返回 critic_loss, actor_loss）
        - 若 critic_vector=False（标量 Q）：使用共享奖励（reward shape=(batch, 1)）
        - 若 critic_vector=True（向量 Q）：奖励应为逐智能体（reward shape=(batch, n_veh)）
        """
        if self.replay_buffer is None or len(self.replay_buffer) < batch_size:
            return None, None

        self.total_it += 1

        # sample
        state_all, action_all, reward, next_state_all, done = self.replay_buffer.sample(batch_size)
        state_all = torch.tensor(state_all, dtype=torch.float32, device=self.device)
        action_all = torch.tensor(action_all, dtype=torch.float32, device=self.device)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)  # shape (batch, 1) or (batch, n_veh)
        next_state_all = torch.tensor(next_state_all, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)  # (batch, 1)

        # target actions
        next_actions_all = self._target_actions(next_state_all)

        # target Q
        if self.critic_vector:
            q1_tgt_all = self.critic1_target(next_state_all, next_actions_all)  # (batch, n_veh)
            q2_tgt_all = self.critic2_target(next_state_all, next_actions_all)
            q_tgt_min_all = torch.min(q1_tgt_all, q2_tgt_all)  # (batch, n_veh)
            # reward_t: (batch, n_veh)
            y_all = reward_t + (1.0 - done) * self.gamma * q_tgt_min_all  # broadcast (batch, n_veh)
            # current Q
            q1_all = self.critic1(state_all, action_all)
            q2_all = self.critic2(state_all, action_all)
            critic_loss = nn.MSELoss()(q1_all, y_all.detach()) + nn.MSELoss()(q2_all, y_all.detach())
        else:
            q1_tgt = self.critic1_target(next_state_all, next_actions_all)  # (batch, 1)
            q2_tgt = self.critic2_target(next_state_all, next_actions_all)  # (batch, 1)
            q_tgt_min = torch.min(q1_tgt, q2_tgt)  # (batch, 1)
            # reward_t: (batch, 1)
            y = reward_t + (1.0 - done) * self.gamma * q_tgt_min
            # current Q
            q1 = self.critic1(state_all, action_all)
            q2 = self.critic2(state_all, action_all)
            critic_loss = nn.MSELoss()(q1, y.detach()) + nn.MSELoss()(q2, y.detach())

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        actor_loss_value = None
        if self.total_it % self.policy_delay == 0:
            if self.actor_shared:
                # 所有智能体使用共享 Actor，构造当前动作并最大化 Q
                actions = []
                for i in range(self.n_veh):
                    obs_i = self._slice_obs(state_all, i)  # (batch, 2)
                    act_i = self.actor(obs_i)              # (batch, 3)
                    actions.append(act_i)
                a_all_prime = torch.cat(actions, dim=1)    # (batch, 3*n_veh)

                if self.critic_vector:
                    q1_all = self.critic1(state_all, a_all_prime)  # (batch, n_veh)
                    actor_loss = -q1_all.mean()  # 对所有智能体的 Q 求均值
                else:
                    q1 = self.critic1(state_all, a_all_prime)       # (batch, 1)
                    actor_loss = -q1.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.detach().cpu().item())
            else:
                # 独立 Actor：逐智能体更新（其他智能体的动作不求梯度）
                actor_losses = []
                for i in range(self.n_veh):
                    actions = []
                    for j in range(self.n_veh):
                        obs_j = self._slice_obs(state_all, j)
                        if j == i:
                            act_j = self.actors[j](obs_j)
                        else:
                            with torch.no_grad():
                                act_j = self.actors[j](obs_j)
                        actions.append(act_j)
                    a_all_prime = torch.cat(actions, dim=1)

                    if self.critic_vector:
                        q1_all = self.critic1(state_all, a_all_prime)  # (batch, n_veh)
                        q1_i = q1_all[:, i:i+1]                        # (batch, 1)
                        actor_loss = -q1_i.mean()
                    else:
                        q1 = self.critic1(state_all, a_all_prime)       # (batch, 1)
                        actor_loss = -q1.mean()

                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    self.actor_optimizers[i].step()
                    actor_losses.append(float(actor_loss.detach().cpu().item()))
                actor_loss_value = float(np.mean(actor_losses))

            # soft update targets
            self._soft_update(self.critic1_target, self.critic1, self.tau)
            self._soft_update(self.critic2_target, self.critic2, self.tau)
            if self.actor_shared:
                self._soft_update(self.actor_target, self.actor, self.tau)
            else:
                for tgt, src in zip(self.actors_target, self.actors):
                    self._soft_update(tgt, src, self.tau)

        return float(critic_loss.detach().cpu().item()), actor_loss_value

    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'n_veh': self.n_veh,
            'actor_shared': self.actor_shared,
            'critic_vector': self.critic_vector,
            'actor_state_dict': None if not self.actor_shared else self.actor.state_dict(),
            'actor_target_state_dict': None if not self.actor_shared else self.actor_target.state_dict(),
            'actors_state_dict': None if self.actor_shared else [a.state_dict() for a in self.actors],
            'actors_target_state_dict': None if self.actor_shared else [a.state_dict() for a in self.actors_target],
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
        }, os.path.join(path, 'matd3.pt'))