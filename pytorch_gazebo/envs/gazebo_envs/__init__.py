from gym.envs.registration import register


def register_gazebo_env(env_id):
    if env_id == 'Alphred-v3':
        register(
            id='Alphred-v3',
            entry_point='gazebo_envs.alphredv3:ALPHREDV3',
            max_episode_steps=1000,
            reward_threshold=6000.0,
        )
    elif env_id:
        # Placeholder for other environment ids.
        print('Please enter in a valid environment id.')
    else:
        print('Please enter in a valid environment id.')

    return None

