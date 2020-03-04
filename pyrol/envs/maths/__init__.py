from gym.envs.registration import register
from pyrol.envs.maths.math_env import *


def register_all_envs():

    register(
        id='PendulumMaths-v0',
        entry_point='pyrol.envs.maths.pendulum:PendulumEnv',
        max_episode_steps=400,
        reward_threshold=6000.0,
    )

    # Add other maths environments here

    return None

