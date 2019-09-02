import numpy as np


class Base(object):
    r"""Base Class for Runners"""
    def step(self, *args, **kwargs):
        raise NotImplementedError


class Runner(Base):
    r"""Runs actions in the environment for experiments"""
    def __init__(self, env, agent, replay_buffer):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer

        self.state = self.env.reset()

    def step(self, initial_state, policy, times=200):
        state = initial_state
        for _ in range(times):
            action = policy(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, next_state, reward, done)

            state = next_state
            if done:
                state = self.env.reset()

        return next_state, reward, done

    def random_policy(self, state=None):
        return self.env.action_space.sample()

    def populate_replay_buffer(self, transitions=10000):
        print(f'Populating replay buffer with {transitions} transitions...')
        self.step(initial_state=self.env.reset(), policy=self.random_policy, times=transitions)

    def agent_select_action(self, state):
        return self.agent.select_action(np.array(state))

    def one_step(self):
        next_state, reward, done = self.step(initial_state=self.state, policy=self.agent_select_action, times=1)
        self.state = next_state
        return reward, done


class TD3Runner(Base):
    r"""Runs actions in the environment for experiments"""
    def __init__(self, env, agent, replay_buffer):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer

        self.state = self.env.reset()

    def step(self, initial_state, policy, times=200):
        state = initial_state
        for _ in range(times):
            action = policy(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, next_state, reward, done)

            state = next_state
            if done:
                state = self.env.reset()

        return next_state, reward, done

    def random_policy(self, state=None):
        return self.env.action_space.sample()

    def populate_replay_buffer(self, transitions=10000):
        print(f'Populating replay buffer with {transitions} transitions...')
        self.step(initial_state=self.env.reset(), policy=self.random_policy, times=transitions)

    def agent_select_action(self, state):
        return self.agent.select_action(np.array(state))

    def one_step(self):
        next_state, reward, done = self.step(initial_state=self.state, policy=self.agent_select_action, times=1)
        self.state = next_state
        return reward, done

