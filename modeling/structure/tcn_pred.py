from logging import Logger

import torch.nn as nn
from torch import nn, device as TorchDevice
from torch.nn.utils import weight_norm

from modeling.structure.pytorch_base import PytorchModelStructure


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, T, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.T = T

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(dropout)

        if T == 2:
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
            self.chomp2 = Chomp1d(padding)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout1d(dropout)

            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                    self.conv2, self.chomp2, self.relu2, self.dropout2)

        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        self.down_sample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        if self.T == 2:
            nn.init.kaiming_normal_(self.conv2.weight)
        if self.down_sample is not None:
            nn.init.kaiming_normal_(self.down_sample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.down_sample is None else self.down_sample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, T, dropout=0.25):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, T=T, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNPred(PytorchModelStructure):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        super(TCNPred, self).__init__(config, logger, device)

        input_size  = config.get('modeling', {}).get('structure', {}).get('input_size')
        num_levels  = config.get('modeling', {}).get('structure', {}).get('num_levels')
        hidden_size = config.get('modeling', {}).get('structure', {}).get('hidden_size')
        kernel_size = config.get('modeling', {}).get('structure', {}).get('kernel_size')
        dropout     = config.get('modeling', {}).get('structure', {}).get('dropout')
        T           = config.get('modeling', {}).get('structure', {}).get('T')
        mlp         = bool(config.get('modeling', {}).get('structure', {}).get('mlp', False))
        self.sig    = bool(config.get('modeling', {}).get('structure', {}).get('sig', True))

        num_channels = [hidden_size] * (num_levels)
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, kernel_size=kernel_size, T=T, dropout=dropout)

        if mlp:
            self.head = nn.Sequential(
                nn.Linear(num_channels[-1], hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, input_size),
            )
        else:
            self.head = nn.Linear(num_channels[-1], input_size)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        # (N, seq, n_bytes)
    
        out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        out = self.head(out)
        if self.sig:
            out = self.sig(out)

        return out
