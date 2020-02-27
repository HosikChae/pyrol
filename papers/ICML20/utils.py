import numpy as np

def env_params(env):
    return (np.prod(env.observation_space.shape), np.prod(env.action_space.shape), env.action_space.high[0].item(), env.action_space.low[0].item())

def remainder(x, y):
    """Remainder of x / y, handles +/- x and +y returns +/-remainder"""
    return x - np.sign(x) * y * (np.abs(x) // y)

def polyak_avg(old_th, new_th, tau=0.005):
    """Polyak averaging ie rolling average"""
    assert tau <= 1., 'Tau needs to be less than or equal to 1.'
    return (1 - tau) * old_th + tau * new_th