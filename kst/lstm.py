# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simple LSTM layers module."""

import torch
from torch import nn


class SLSTM(nn.Module):
    """ Simple LSTM layer.

    Args:
        dimension: int, the dimension of the input and output.
        num_layers: int, the number of layers in the LSTM.
        skip: bool, whether to add the input to the output.
        bidirectional: bool, whether to use bidirectional LSTM.
    """
    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True, bidirectional: bool=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers, bidirectional=bidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)

        if self.bidirectional:
            x = x.repeat(1, 1, 2)

        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        
        return y
