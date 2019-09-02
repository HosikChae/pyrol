"""Functions to help when debugging"""
import numpy as np

###########
# PyTorch #
###########

def is_cuda(parameter):
    try:
        return parameter.is_cuda
    except AttributeError:
        return next(parameter.parameters()).is_cuda


##########################
# Reinforcement Learning #
##########################


def evaluate_policy(policy, env, eps=100, render=False):
    avg_reward = 0.
    for i in range(eps):
        state = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = policy.select_action(np.array(state), noise=0)
            state, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eps

    print(f'Evaluation Episodes: {eps} Average Score: {avg_reward}')
    return avg_reward

