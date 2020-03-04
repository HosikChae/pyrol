import torch

#############
# DEBUGGING #
#############


def is_cuda(parameter):
    try:
        return parameter.is_cuda
    except AttributeError:
        return next(parameter.parameters()).is_cuda


####################
# DATA CONVERSIONS #
####################


def return_self(x):
    return x


def np2torch_float(x):
    return torch.from_numpy(x).float()

