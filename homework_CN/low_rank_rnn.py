"""Low-rank RNN model for the group coursework.

Created by Cai Chunqiu on May 7, 12:17.
"""

import torch
import torch.nn as nn


def ortho_init(tensor, scale=1.0):
    """Orthogonal initializer matching the style used in the main project."""
    if tensor.ndim < 2:
        nn.init.zeros_(tensor)
        return tensor
    nn.init.orthogonal_(tensor)
    tensor.mul_(scale)
    return tensor


class LowRankRNN(nn.Module):
    """Continuous-time low-rank RNN for the 3-context Go/No-Go task.

    The recurrent matrix is represented as W_rec = U V^T. The forward pass uses
    the same discretized dynamics as models/ct_lr_rnn.py, but removes training
    stages, noise switches, and fixed batch-size assumptions.
    """

    def __init__(
        self,
        input_size=5,
        hidden_size=512,
        output_size=1,
        rank=2,
        alpha=0.2,
        train_h0=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rank = rank
        self.alpha = alpha

        self.h0 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=train_h0)
        self.W_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(hidden_size, rank))
        self.V = nn.Parameter(torch.empty(hidden_size, rank))
        self.W_out = nn.Parameter(torch.empty(output_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            ortho_init(self.W_in)
            ortho_init(self.U)
            ortho_init(self.V)
            ortho_init(self.W_out, scale=1.0)

    def forward(self, inputs, h0=None, return_states=False):
        """Run the RNN over a full trial.

        Args:
            inputs: tensor with shape [batch, time, input_size].
            h0: optional initial state with shape [batch, hidden_size].
            return_states: if True, also return hidden states [batch, time, hidden].
        """
        batch_size, seq_len, _ = inputs.shape
        if h0 is None:
            h = self.h0.expand(batch_size, -1)
        else:
            h = h0

        outputs = []
        states = []
        for t in range(seq_len):
            rates = torch.tanh(h)
            recurrent = (rates @ self.V) @ self.U.t()
            recurrent = recurrent / self.hidden_size
            input_drive = inputs[:, t, :] @ self.W_in.t()
            h = h + self.alpha * (-h + recurrent + input_drive)
            rates = torch.tanh(h)
            y = rates @ self.W_out.t()
            outputs.append(y)
            if return_states:
                states.append(h)

        outputs = torch.stack(outputs, dim=1)
        if return_states:
            return outputs, torch.stack(states, dim=1)
        return outputs
