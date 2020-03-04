import numpy as np

def remainder(x, y):
    """Remainder of x / y, handles +/- x and +y returns +/-remainder"""
    return x - np.sign(x) * y * (np.abs(x) // y)

def polyak_avg(old_th, new_th, tau=0.005):
    """Polyak averaging ie rolling average"""
    assert tau <= 1., 'Tau needs to be less than or equal to 1.'
    return (1 - tau) * old_th + tau * new_th