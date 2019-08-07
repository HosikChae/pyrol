from gym.envs.registration import register


# TODO: Fix workaround for buggy registration
def register_alphredv3():
    register(
        id='alphred-v3',
        entry_point='gazebo_envs.alphredv3:ALPHREDV3',
        max_episode_steps=1000,  # TODO: Figure out best parameters
        reward_threshold=6000.0,
        nondeterministic=True,
        kwargs={},
    )

    return None

