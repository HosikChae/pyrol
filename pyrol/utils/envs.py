import numpy as np

def env_params(env):
    """Gets parameters needed to setup environment"""
    # TODO: Need to generalize max and min actions, fix actor such that the vector multiplication on gpu side and max individul
    return (np.prod(env.observation_space.shape), np.prod(env.action_space.shape), env.action_space.high[0].item(), env.action_space.low[0].item())