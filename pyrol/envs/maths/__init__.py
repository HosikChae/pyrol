from gym.envs.registration import register


def register_all_envs():

    register(
        id='Pendulum-maths0',
        entry_point='pyrol.envs.maths.pendulum:PendulumEnv',
        max_episode_steps=1000,
        reward_threshold=6000.0,
    )

    # Add other maths environments here

    return None

