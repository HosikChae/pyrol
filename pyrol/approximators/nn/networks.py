import torch
import torch.nn as nn


def linear_block(dims, layer_activations):
    assert isinstance(dims, (tuple, list)) and isinstance(layer_activations, (tuple, list)), TypeError
    assert len(dims) == len(layer_activations) + 1, 'Need to have `layers - 1` activations. No type is `linear`.'
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()],
        ['tanh', nn.Tanh()],
        ['linear', ReturnSelf()],
    ])

    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activations[layer_activations[dims.index(in_dim)]])

    return nn.Sequential(*layers)


class Base(object):
    r"""Base Class for Neural Network"""
    def forward(self):
        """Forward Pass defined in PyTorch"""
        raise NotImplementedError


class ReturnSelf(nn.Module, Base):
    def __init__(self):
        super(ReturnSelf, self).__init__()

    def forward(self, x):
        return x


class Actor(nn.Module, Base):
    def __init__(self, state_dim, action_dim, max_action, hidden_dims=(400, 300)):
        super(Actor, self).__init__()
        dims = (state_dim,) + hidden_dims + (action_dim,)
        self.linear_block = linear_block(dims=dims, layer_activations=('relu', 'relu', 'tanh'))
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.linear_block(x)


class TD3Critic(nn.Module, Base):
    def __init__(self, state_dim, action_dim, hidden_dims=(400, 300)):
        super(TD3Critic, self).__init__()
        dims = (state_dim + action_dim,) + hidden_dims + (1,)

        self.q1 = linear_block(dims=dims, layer_activations=('relu', 'relu', 'linear'))
        self.q2 = linear_block(dims=dims, layer_activations=('relu', 'relu', 'linear'))

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        return self.q1(xu), self.q2(xu)


