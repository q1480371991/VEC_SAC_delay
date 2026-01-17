import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import Actor, Critic
from network_rnn import RNNActor

def get_device(use_gpu: bool = True, device_idx: int = 0):
    if use_gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{device_idx}")
    return torch.device("cpu")

class MultiCritic(nn.Module):
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
    MATD3 with optional RNN actors (GRU/LSTM):
    - critic_vector: False -> scalar Q; True -> per-agent Q vector
    - actor_shared: True -> one shared actor; False -> per-agent independent actors
    - rnn_type: 'none' | 'gru' | 'lstm'
      在线执行时维护隐状态；训练/目标动作时使用零隐状态（简化版，后续可扩展序列回放）
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
        rnn_type: str = 'gru',
        rnn_hidden_size: int = 128,
    ):
        assert rnn_type in ('none', 'gru', 'lstm'), "rnn_type must be 'none', 'gru', or 'lstm'"
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
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size

        self.replay_buffer = replay_buffer
        self.device = get_device(use_gpu=use_gpu, device_idx=device_idx)

        # Actors
        if self.rnn_type == 'none':
            ActorImpl = lambda: Actor(self.obs_dim, self.act_dim, self.hidden_dim, self.action_range)
        else:
            ActorImpl = lambda: RNNActor(self.obs_dim, self.act_dim, self.rnn_hidden_size, self.action_range, cell_type=self.rnn_type)

        if self.actor_shared:
            self.actor = ActorImpl().to(self.device)
            self.actor_target = ActorImpl().to(self.device)
            self.actors = None
            self.actors_target = None
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            # 在线隐状态：每智能体一个
            self.hidden_states = [self._init_hidden(batch_size=1) for _ in range(self.n_veh)]
            self.hidden_states_target = [self._init_hidden(batch_size=1) for _ in range(self.n_veh)]
        else:
            self.actors = nn.ModuleList([ActorImpl().to(self.device) for _ in range(self.n_veh)])
            self.actors_target = nn.ModuleList([ActorImpl().to(self.device) for _ in range(self.n_veh)])
            self.actor = None
            self.actor_target = None
            self.actor_optimizers = [optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]
            # 在线隐状态：每智能体一个
            self.hidden_states = [self._init_hidden(batch_size=1) for _ in range(self.n_veh)]
            self.hidden_states_target = [self._init_hidden(batch_size=1) for _ in range(self.n_veh)]

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

    def _init_hidden(self, batch_size=1):
        if self.rnn_type == 'none':
            return None
        if self.rnn_type == 'gru':
            return torch.zeros(batch_size, self.rnn_hidden_size, dtype=torch.float32, device=self.device)
        else:
            h = torch.zeros(batch_size, self.rnn_hidden_size, dtype=torch.float32, device=self.device)
            c = torch.zeros(batch_size, self.rnn_hidden_size, dtype=torch.float32, device=self.device)
            return (h, c)

    def reset_hidden(self, batch_size=1):
        if self.rnn_type == 'none':
            return
        self.hidden_states = [self._init_hidden(batch_size) for _ in range(self.n_veh)]
        self.hidden_states_target = [self._init_hidden(batch_size) for _ in range(self.n_veh)]

    def _hard_update(self, target: nn.Module, source: nn.Module):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(s_param.data)

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - tau) + s_param.data * tau)

    def _slice_obs(self, state_all: torch.Tensor, i: int) -> torch.Tensor:
        if state_all.dim() == 1:
            n = self.n_veh
            return torch.stack([state_all[i], state_all[n + i]], dim=0)
        else:
            n = self.n_veh
            return torch.stack([state_all[:, i], state_all[:, n + i]], dim=1)

    def get_actions(self, state_all_np: np.ndarray, deterministic: bool = False, explore_noise: float = 0.0):
        """
        在线执行用：维护并更新隐状态
        输入: (2*n_veh,)
        输出: (3*n_veh,) in [-1,1]
        """
        state_all = torch.tensor(state_all_np, dtype=torch.float32, device=self.device).flatten()
        actions = []
        if self.actor_shared:
            for i in range(self.n_veh):
                obs_i = self._slice_obs(state_all, i).unsqueeze(0)  # (1, 2)
                if self.rnn_type == 'none':
                    act_i = self.actor(obs_i).squeeze(0)
                else:
                    act_i, new_h = self.actor(obs_i, self.hidden_states[i])
                    self.hidden_states[i] = new_h  # update hidden
                    act_i = act_i.squeeze(0)
                actions.append(act_i)
        else:
            for i in range(self.n_veh):
                obs_i = self._slice_obs(state_all, i).unsqueeze(0)
                if self.rnn_type == 'none':
                    act_i = self.actors[i](obs_i).squeeze(0)
                else:
                    act_i, new_h = self.actors[i](obs_i, self.hidden_states[i])
                    self.hidden_states[i] = new_h
                    act_i = act_i.squeeze(0)
                actions.append(act_i)
        action_all = torch.cat(actions, dim=0)

        if not deterministic and explore_noise > 0.0:
            noise = torch.randn_like(action_all) * explore_noise
            action_all = torch.clamp(action_all + noise, -1.0, 1.0)

        return action_all.detach().cpu().numpy()

    def _target_actions(self, next_state_all: torch.Tensor):
        """
        训练/目标网络用：使用零隐状态（简化）
        next_state_all: (batch, 2*n_veh)
        返回: (batch, 3*n_veh)
        """
        actions = []
        if self.actor_shared:
            for i in range(self.n_veh):
                obs_i = self._slice_obs(next_state_all, i)  # (batch, 2)
                if self.rnn_type == 'none':
                    act_i = self.actor_target(obs_i)  # (batch, 3)
                else:
                    h0 = self._init_hidden(batch_size=obs_i.shape[0])
                    act_i, _ = self.actor_target(obs_i, h0)
                actions.append(act_i)
        else:
            for i in range(self.n_veh):
                obs_i = self._slice_obs(next_state_all, i)
                if self.rnn_type == 'none':
                    act_i = self.actors_target[i](obs_i)
                else:
                    h0 = self._init_hidden(batch_size=obs_i.shape[0])
                    act_i, _ = self.actors_target[i](obs_i, h0)
                actions.append(act_i)
        next_actions_all = torch.cat(actions, dim=1)  # (batch, 3*n_veh)

        # policy noise
        noise = torch.clamp(torch.randn_like(next_actions_all) * self.policy_noise, -self.noise_clip, self.noise_clip)
        next_actions_all = torch.clamp(next_actions_all + noise, -1.0, 1.0)
        return next_actions_all

    def update(self, batch_size: int):
        if self.replay_buffer is None or len(self.replay_buffer) < batch_size:
            return None, None

        self.total_it += 1
        # sample
        s_all, a_all, r, s_next_all, done = self.replay_buffer.sample(batch_size)
        s_all = torch.tensor(s_all, dtype=torch.float32, device=self.device)
        a_all = torch.tensor(a_all, dtype=torch.float32, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)           # (batch, 1) or (batch, n_veh)
        s_next_all = torch.tensor(s_next_all, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)     # (batch, 1)

        next_actions_all = self._target_actions(s_next_all)

        if self.critic_vector:
            q1_tgt_all = self.critic1_target(s_next_all, next_actions_all)
            q2_tgt_all = self.critic2_target(s_next_all, next_actions_all)
            q_tgt_min_all = torch.min(q1_tgt_all, q2_tgt_all)
            y_all = r + (1.0 - done) * self.gamma * q_tgt_min_all
            q1_all = self.critic1(s_all, a_all)
            q2_all = self.critic2(s_all, a_all)
            critic_loss = nn.MSELoss()(q1_all, y_all.detach()) + nn.MSELoss()(q2_all, y_all.detach())
        else:
            q1_tgt = self.critic1_target(s_next_all, next_actions_all)
            q2_tgt = self.critic2_target(s_next_all, next_actions_all)
            q_tgt_min = torch.min(q1_tgt, q2_tgt)
            y = r + (1.0 - done) * self.gamma * q_tgt_min
            q1 = self.critic1(s_all, a_all)
            q2 = self.critic2(s_all, a_all)
            critic_loss = nn.MSELoss()(q1, y.detach()) + nn.MSELoss()(q2, y.detach())

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        actor_loss_value = None
        if self.total_it % self.policy_delay == 0:
            if self.actor_shared:
                actions = []
                for i in range(self.n_veh):
                    obs_i = self._slice_obs(s_all, i)
                    if self.rnn_type == 'none':
                        act_i = self.actor(obs_i)
                    else:
                        h0 = self._init_hidden(batch_size=obs_i.shape[0])  # 零隐状态
                        act_i, _ = self.actor(obs_i, h0)
                    actions.append(act_i)
                a_all_prime = torch.cat(actions, dim=1)
                if self.critic_vector:
                    q1_all = self.critic1(s_all, a_all_prime)
                    actor_loss = -q1_all.mean()
                else:
                    q1 = self.critic1(s_all, a_all_prime)
                    actor_loss = -q1.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_loss_value = float(actor_loss.detach().cpu().item())
            else:
                actor_losses = []
                for i in range(self.n_veh):
                    actions = []
                    for j in range(self.n_veh):
                        obs_j = self._slice_obs(s_all, j)
                        if j == i:
                            if self.rnn_type == 'none':
                                act_j = self.actors[j](obs_j)
                            else:
                                h0 = self._init_hidden(batch_size=obs_j.shape[0])
                                act_j, _ = self.actors[j](obs_j, h0)
                        else:
                            with torch.no_grad():
                                if self.rnn_type == 'none':
                                    act_j = self.actors[j](obs_j)
                                else:
                                    h0 = self._init_hidden(batch_size=obs_j.shape[0])
                                    act_j, _ = self.actors[j](obs_j, h0)
                        actions.append(act_j)
                    a_all_prime = torch.cat(actions, dim=1)
                    if self.critic_vector:
                        q1_all = self.critic1(s_all, a_all_prime)
                        q1_i = q1_all[:, i:i+1]
                        actor_loss = -q1_i.mean()
                    else:
                        q1 = self.critic1(s_all, a_all_prime)
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
            'rnn_type': self.rnn_type,
            'rnn_hidden_size': self.rnn_hidden_size,
            'actor_state_dict': None if not self.actor_shared else self.actor.state_dict(),
            'actor_target_state_dict': None if not self.actor_shared else self.actor_target.state_dict(),
            'actors_state_dict': None if self.actor_shared else [a.state_dict() for a in self.actors],
            'actors_target_state_dict': None if self.actor_shared else [a.state_dict() for a in self.actors_target],
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
        }, os.path.join(path, 'matd3.pt'))