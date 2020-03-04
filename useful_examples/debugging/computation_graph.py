"""Visualizing Computation Graphs
For more examples see the link below:
https://github.com/szagoruyko/pytorchviz/blob/master/examples.ipynb
"""

import torch
from torch import nn
from torchviz import make_dot

# Example 1
model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)

dot = make_dot(model(x), params=dict(model.named_parameters()))
dot.render('test1', view=True)

# Example 2
a = torch.randn(1, requires_grad=True, dtype=torch.float)
b = torch.randn(1, requires_grad=True, dtype=torch.float)

yhat = a + b * 10
error = 2 - yhat
loss = (error ** 2).mean()
dot = make_dot(yhat)
dot.render('test2', view=True)

# TODO: make it such that the graphs aren't saved and can be dynamic