import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNActor(nn.Module):
    """
    RNN Actor（GRU/LSTM）:
    - 输入: state_i (batch, obs_dim)
    - 隐状态: GRU -> (batch, hidden_size) ; LSTM -> (h, c) 各为 (batch, hidden_size)
    - 输出: action_i (batch, act_dim), 经过 tanh 映射到 [-1, 1] 后再 * action_range
    """
    def __init__(self, obs_dim, act_dim, hidden_size, action_range=1.0, cell_type='gru', init_w=3e-3):
        super().__init__()
        assert cell_type in ('gru', 'lstm'), "cell_type must be 'gru' or 'lstm'"
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.action_range = action_range
        self.cell_type = cell_type

        if cell_type == 'gru':
            self.rnn_cell = nn.GRUCell(obs_dim, hidden_size)
        else:
            self.rnn_cell = nn.LSTMCell(obs_dim, hidden_size)

        # MLP head from hidden -> action
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, act_dim)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, hidden):
        """
        state: (batch, obs_dim)
        hidden:
          - GRU: (batch, hidden_size)
          - LSTM: tuple(h, c), each (batch, hidden_size)
        返回: action (batch, act_dim), new_hidden (同上)
        """
        if self.cell_type == 'gru':
            h = self.rnn_cell(state, hidden)  # (batch, hidden_size)
        else:
            h, c = self.rnn_cell(state, hidden)  # each (batch, hidden_size)

        x = F.relu(self.fc1(h if self.cell_type == 'gru' else h))
        act = torch.tanh(self.fc2(x)) * self.action_range

        if self.cell_type == 'gru':
            return act, h
        else:
            return act, (h, c)

    def init_hidden(self, batch_size, device):
        if self.cell_type == 'gru':
            return torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=device)
        else:
            h = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=device)
            c = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=device)
            return (h, c)