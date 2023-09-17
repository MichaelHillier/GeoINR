import torch
import numpy as np
import math
from torch import nn
import torch.nn.functional as F
import geoinr.args as args
from geoinr.model.mlp.layers import Perceptron, PositionalEncoding


def get_activation_function_(params: args.argparse.Namespace):
    # ['relu', 'splus, 'elu', 'gelu', 'mish', 'silu', 'selu', 'prelu', 'sin']
    name = params.activation
    if name == 'relu':
        return F.relu
    elif name == 'splus':
        return nn.Softplus(beta=params.beta)
    elif name == 'elu':
        return nn.ELU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'mish':
        return nn.Mish()
    elif name == 'silu':
        return nn.SiLU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'prelu':
        return nn.PReLU()
    else:
        return False


class SimpleMLP(nn.Module):
    def __init__(self, in_features, params: args.argparse.Namespace, out_features):
        super().__init__()

        hidden_features = params.embed_dim
        hidden_layers = params.num_hidden_layers
        concat = params.concat
        activation = get_activation_function_(params)

        self.net = []
        self.net.append(Perceptron(in_features, hidden_features, activation, concat=concat))

        if concat:
            h_dim_concat = in_features + hidden_features
            for i in range(hidden_layers):
                self.net.append(Perceptron(h_dim_concat, h_dim_concat, activation, concat=concat))
                h_dim_concat *= 2

            self.net.append(Perceptron(h_dim_concat, out_features, activation, is_last=True))
        else:
            for i in range(hidden_layers):
                self.net.append(Perceptron(hidden_features, hidden_features, activation))

            self.net.append(Perceptron(hidden_features, out_features, activation, is_last=True))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


class PositionalMLP(nn.Module):
    def __init__(self, in_features, params: args.argparse.Namespace, out_features):
        super().__init__()
        hidden_features = params.embed_dim
        hidden_layers = params.num_hidden_layers
        concat = params.concat
        activation = get_activation_function_(params)

        self.net = []
        pos_layer = PositionalEncoding(in_features, 50, 0.5)
        #pos_layer = PosEncodingNeRF(in_features)

        self.net.append(pos_layer)
        self.net.append(Perceptron(pos_layer.out_features, hidden_features, activation, concat=concat))

        if concat:
            h_dim_concat = in_features + hidden_features
            for i in range(hidden_layers):
                self.net.append(Perceptron(h_dim_concat, h_dim_concat, activation, concat=concat))
                h_dim_concat *= 2

            self.net.append(Perceptron(h_dim_concat, out_features, activation, is_last=True))
        else:
            for i in range(hidden_layers):
                self.net.append(Perceptron(hidden_features, hidden_features, activation))

            self.net.append(Perceptron(hidden_features, out_features, activation, is_last=True))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords


class SeriesMLP(nn.Module):
    def __init__(self, in_features, params: args.argparse.Namespace, out_features, weights_file=None):
        super().__init__()

        self.n_series = out_features
        self.series_mlp = nn.ModuleList()
        for i in range(self.n_series):
            m = SimpleMLP(in_features, params, 1)

            if weights_file is not None:
                m.load_state_dict(torch.load(weights_file))
            self.series_mlp.append(m)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = torch.cat([self.series_mlp[i].net(coords) for i in range(self.n_series)], dim=1)
        return output, coords


class SingleSkip(nn.Module):
    """
    This skip model features: Single skip connection adds x ([x, y, z]) projected to 1D to output
    of a single skip layer. Can't have multiple skip connections in this model, since the output of the
    skip layer is 1D, if output scalar field is negative everywhere then relu activation sets
    everything to zero, there all outputs will be same (since input is the same).
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        self.proj = nn.Linear(in_features, out_features)
        self.activation = F.relu

        self.net = []
        self.net.append(Perceptron(in_features, hidden_features))
        for j in range(hidden_layers):
            self.net.append(Perceptron(hidden_features, hidden_features))
        self.net.append(Perceptron(hidden_features, out_features, is_last=True))
        self.net = nn.Sequential(*self.net)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight, gain=1.414)
        nn.init.zeros_(self.proj.bias)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        x_skip = self.proj(coords)
        x = self.net(coords)
        x = x + x_skip
        return x, coords


class Skip(nn.Module):
    """
    This skip model features: 1st skip connections adds x ([x, y, z]) projected to {hidden_features}D
    to output of skip layer. Final linear layer projects {hidden_features}D output to 1D (out_features).
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features, nskips):
        super().__init__()
        self.n_skips = nskips

        self.proj = nn.Linear(in_features, hidden_features)
        self.pred = nn.Linear(hidden_features, out_features)
        self.activation = F.relu
        # self.activation = nn.SELU()
        # self.activation = nn.PReLU()

        self.layers = nn.ModuleList()
        in_dim = in_features
        for i in range(self.n_skips):
            net = []
            net.append(Perceptron(in_dim, hidden_features))
            for j in range(hidden_layers):
                if j == hidden_layers - 1:
                    is_last = True
                else:
                    is_last = False
                net.append(Perceptron(hidden_features, hidden_features, is_last=is_last))
            net = nn.Sequential(*net)
            self.layers.append(net)
            in_dim = hidden_features

        self.init_weights()

    def init_weights(self):
        gain = nn.init.calculate_gain('relu')
        # gain = nn.init.calculate_gain('selu')
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.pred.weight, gain=gain)
        nn.init.zeros_(self.pred.bias)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        x = coords
        for i, layer in enumerate(self.layers):
            if i == 0:
                x_skip = self.proj(x)
            else:
                x_skip = x
            x = layer(x)
            x = self.activation(x + x_skip)

        x = self.pred(x)

        return x, coords


class Residual(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, n_residuals):
        super().__init__()

        self.layers = nn.ModuleList()
        in_dim = in_features
        for i in range(n_residuals):
            net = []
            net.append(Perceptron(in_dim, hidden_features))
            for i in range(hidden_layers):
                net.append(Perceptron(hidden_features, hidden_features))
            net.append(Perceptron(hidden_features, out_features, is_last=True))
            in_dim = out_features
            net = nn.Sequential(*net)
            self.layers.append(net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        x = coords
        for i, layer in enumerate(self.layers):
            if i == 0:
                s = layer(coords)
            else:
                s = s + layer(s)
                # s += layer(s)
        return s, coords


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 1
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        # coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        #return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

        return coords_pos_enc.reshape(coords.shape[0], self.out_dim)