# TODO: get rid of gym dependency should be easy for these environments

def gym_register(sim):
    if sim == 'gazebo':
        from .gazebo import register_all_envs
        register_all_envs()
    elif sim == 'maths':
        from .maths import register_all_envs
        register_all_envs()
    else:
        raise Exception('Please enter in a valid simulator.')

    return None

