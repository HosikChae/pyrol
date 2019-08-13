from gym.envs.registration import register


def register_all_envs():

    register(
        id='Alphred-v3',
        entry_point='gazebo.alphredv3:ALPHREDV3',
        max_episode_steps=1000,
        reward_threshold=6000.0,
    )

    # Add other gazebo gym environments here

    return None

