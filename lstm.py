
# -*- coding:utf8 -*-

import torch
from torch import nn
from collections import defaultdict


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = 2
        self.hidden_size = 16

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )

        self.l = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.input_size)
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (N, L, 2) -> (L, N, 2)
        x = self.lstm(x)[0][-1]  # (L, N, 2) -> (N, 64)
        x = self.l(x)  # (N, 64) -> (N, 2)
        return x
