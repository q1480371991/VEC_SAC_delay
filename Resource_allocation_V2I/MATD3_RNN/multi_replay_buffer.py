import random
import numpy as np

class MultiReplayBuffer:
    """
    多智能体回放缓冲区（支持共享奖励或逐智能体奖励）
    存储条目：(state_all, action_all, reward, next_state_all, done)
    - state_all: shape (2*n_veh,)
    - action_all: shape (3*n_veh,)
    - reward: shape (1,) for shared_global; shape (n_veh,) for per-agent
    - next_state_all: shape (2*n_veh,)
    - done: scalar (0/1)
    """
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state_all, action_all, reward, next_state_all, done):
        state_all = np.asarray(state_all, dtype=np.float32).flatten()
        action_all = np.asarray(action_all, dtype=np.float32).flatten()
        reward = np.asarray(reward, dtype=np.float32).reshape(-1)  # (1,) or (n_veh,)
        next_state_all = np.asarray(next_state_all, dtype=np.float32).flatten()
        done = float(done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_all, action_all, reward, next_state_all, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_all, action_all, reward, next_state_all, done = zip(*batch)

        state_all = np.stack(state_all, axis=0)
        action_all = np.stack(action_all, axis=0)
        # 奖励长度可能是 1 或 n_veh，保持原形状堆叠
        reward = np.stack(reward, axis=0)
        next_state_all = np.stack(next_state_all, axis=0)
        done = np.array(done, dtype=np.float32).reshape(-1, 1)

        return state_all, action_all, reward, next_state_all, done

    def __len__(self):
        return len(self.buffer)