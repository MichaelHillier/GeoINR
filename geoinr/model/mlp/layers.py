import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class Perceptron(nn.Module):
    def __init__(self, in_features, out_features, activation, bias=True, is_last=False, concat=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bias = bias
        self.is_last = is_last
        self.concat = concat
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain('relu')
        # gain = nn.init.calculate_gain('selu')
        # nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='selu')
        #nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='linear')
        # if self.is_last:
        #     self.linear.weight.data.fill_(1.0 / 3.0)
        # else:
        #     nn.init.normal_(self.linear.weight, mean=0.0, std=np.sqrt(2 / self.out_features))
        # if self.bias:
        #     nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        if self.is_last:
            return self.linear(input)
        else:
            if self.concat:
                return torch.cat((input, self.activation(self.linear(input))), 1)
                # return self.activation(torch.cat((input, self.linear(input)), 1))  # no real diff this or above
            else:
                return self.activation(self.linear(input))


class PositionalEncoding(nn.Module):
    def __init__(self, in_features, num_frequencies, std=1.0):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.std = std
        self.out_features = self.num_frequencies * 2
        self.frequency_spectrum = nn.Linear(self.in_features, self.num_frequencies)
        self.pi = 3.141592653589793
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.frequency_spectrum.weight, 0, self.std)
        nn.init.zeros_(self.frequency_spectrum.bias)

    def forward(self, x):
        return torch.cat((torch.cos(2.0*self.pi*self.frequency_spectrum(x)),
                          torch.sin(2.0*self.pi*self.frequency_spectrum(x))), dim=1)

