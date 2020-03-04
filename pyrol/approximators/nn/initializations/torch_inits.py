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

    Notes
    Early proposed method for initialization of deep neural network architectures

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
    Use when using ReLU, Leaky ReLU, Swish, or any similar activation function in layer
    If use torch.nn.Linear use mode='fan_in'
    If use self-defined linear layer use mode='fan_out'

    Notes
    -Takes into account the ReLU activation function following each matrix multiplication
    -Mostly similar to Xavier Initialization but updated for ReLU
    -Works well in practice for ReLU
    -If architecture more complicated (not ReLU), it may not be able to keep standard deviation around 1
    -`fan_in` refers to the size of the input vector of layer eg number of neurons in previous layer
    -`fan_in` preserves magnitude and variance of the weights in the feed forward phase
    -`fan_in` use this mode when creating weights implicitly with torch.nn.Linear
    -`fan_out` refers to the size of the output vector of layer eg number of neurons in this layer
    -`fan_out` preserves the magnitude during backpropagation
    -`fan_out` use this mode if self defined layer with random matrix initialization
    -torch.nn.Linear implicitly transposes weight matrix, that's why if self defined `fan_in` and `fan_out` modes exist

    Math
    Uniform Distribution, U(-bound, bound):
        bound = (6/ ((1 + a ** 2) x fan_in)) ** 0.5
    Normal Distribution, N(0, std ** 2)
        std = (2/ ((1 + a ** 2) x fan_in)) ** 0.5

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


def orthogonal_init(tensor, gain=1.):
    r"""Semi-Orthogonal Matrix Initialization Scheme for Deep Models
    Paper: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    Authors: Saxe, A. et al.
    Year: 2013

    :param tensor: tensor, tensor of weights eg torch.nn.Conv2d(...).weight
    :param gain: float, multiplied to tensor after initialization
    :return: None
    """
    nn.init.orthogonal_(tensor, gain=gain)


def sparse_init(tensor, sparsity, std=0.01):
    r"""Sparse Initialization of 2D Tensor, Non-Zero Elements From Normal Distribution N(0, std)
    Paper: Deep learning via Hessian-free optimization
    Authors: Martens, J.
    Year: 2010

    :param tensor: tensor, tensor of weights eg torch.nn.Conv2d(...).weight
    :param sparsity: float, fraction of elements in each column to be set to zero
    :param std: float, standard deviation of the normal distribution for non-zero elements
    :return: None
    """
    nn.init.sparse_(tensor, sparsity, std=std)

# TODO: Look into the following inits
# torch.nn.init.dirac_(tensor) for CNN
# implement LSUV ~ supposed to perform better than most initializations
