import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, trainable_omega=True):
        super().__init__()

        if not trainable_omega:
            self.omega_0 = omega_0
        else:
            self.omega_0 = nn.Parameter(torch.Tensor(1).fill_(omega_0))

        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                dim = 1
            else:
                dim = self.in_features
            w_std = (1 / dim) if self.is_first else (np.sqrt(6 / dim) / float(self.omega_0))
            self.linear.weight.uniform_(-w_std, w_std)
            self.linear.bias.uniform_(-w_std, w_std)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Perceptron(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_last=False, concat=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bias = bias
        self.is_last = is_last
        self.concat = concat
        #self.activation = F.relu
        self.activation = nn.Softplus(beta=100)
        #self.activation = nn.ELU()
        #self.activation = nn.GELU()
        #self.activation = nn.Mish()
        #self.activation = nn.SiLU()
        #self.activation = nn.SELU()
        #self.activation = nn.PReLU()

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

