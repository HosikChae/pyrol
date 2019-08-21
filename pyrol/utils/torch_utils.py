import torch.nn as nn

"""
TENSOR INITIALIZATIONS
"""

"""Deep Feedforward Layer Initializations"""


def glorot_init(tensor, gain=1.0, distribution='uniform'):
    r"""Glorot Initialization for Deep Neural Network Architectures
    Paper: `Understanding the difficulty of training deep feedforward neural networks`
    Authors: Glorot, X. and Bengio, Y.
    Year: 2010

    Monikers: Glorot Initialization, Xavier Initialization

    :param gain: float, affects the spread or standard deviation of the distribution proportionally
    :param distribution: str, 'uniform' or 'normal'
    :param tensor: tensor, tensor of weights eg torch.nn.Conv2d(...).weight
    :return: None
    """
    if distribution == 'uniform':
        nn.init.xavier_uniform_(tensor, gain=gain)
    else:
        nn.init.xavier_normal_(tensor, gain=gain)


def kaiming_init(tensor, a=0., mode='fan_in', nonlinearity='leaky_relu', distribution='uniform'):
    r"""Kaiming Initialization for Deep Neural Network Architectures
    Paper: 'Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification`
    Authors: He, K. et al.
    Year: 2015

    Monikers: Kaiming Initialization, He Initialization

    TL;DR


    Notes:

    :param nonlinearity: str, activation function
    :param mode: fan_in or fan_out, dim of tensor input or output
    :param a: float, affects the spread or standard deviation inversely
    :param distribution: str, 'uniform' or 'normal'
    :param tensor: tensor, tensor of weights eg torch.nn.Conv2d(...).weight
    :return: None
    """
    if distribution == 'uniform':
        nn.init.xavier_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.xavier_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity)
