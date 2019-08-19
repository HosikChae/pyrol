import numpy as np


def remainder(x, y):
    """Remainder of x / y"""
    return x - x * (np.abs(x) // y)

