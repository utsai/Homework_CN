"""LSTM baseline for the group coursework.

Created by Cai Chunqiu on May 7, 12:31.
"""

import torch
import torch.nn as nn


class LSTMRNN(nn.Module):
    """LSTM baseline with the same input/output contract as the RNN models."""

    def __init__(
        self,
        input_size=5,
        hidden_size=512,
        output_size=1,
        num_layers=1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, inputs, h0=None, return_states=False):
        states, _ = self.lstm(inputs, h0)
        outputs = self.readout(states)
        if return_states:
            return outputs, states
        return outputs
