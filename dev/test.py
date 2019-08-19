from pyrol.to_be_deleted.deep_rl import *


def ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 1e5  # 1e6
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # select_device(-1)
    select_device(0)

    # from pyrol.envs import gym_register
    # sim = 'gazebo'
    # env_id = 'Alphred-v3'
    # gym_register(sim)
    # ppo_continuous(game=env_id, sim=sim)

    from pyrol.envs import gym_register
    sim = 'maths'
    env_id = 'PendulumMaths-v0'
    gym_register(sim)
    ppo_continuous(game='PendulumMaths-v0')

