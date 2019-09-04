import numpy as np

##########################
# Reinforcement Learning #
##########################


def evaluate_policy(policy, env, max_steps=200, eps=100, render=False):
    avg_reward = 0.
    for i in range(eps):
        state = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            if render:
                env.render()
            action = policy.select_action(np.array(state), noise=False)
            state, reward, done, _ = env.step(action)
            step += 1
            avg_reward += reward

    avg_reward /= eps

    print(f'Evaluation Episodes: {eps} Average Score: {avg_reward}')
    return avg_reward

