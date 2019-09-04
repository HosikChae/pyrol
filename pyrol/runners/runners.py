import numpy as np
from pyrol.utils.metrics import evaluate_policy


class Base(object):
    r"""Base Class for Runners
    Runners should be treated as the necessary interface between the agent, replay buffer, and the environment.
    There should only be one pass of the components here and not into each other.
    """
    def step(self, *args, **kwargs):
        raise NotImplementedError


class Runner(Base):
    r"""Runs actions in the environment for experiments"""
    def __init__(self, env, agent, replay_buffer):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer

        self.state = self.env.reset()

    def step(self, initial_state, policy, steps=200):
        state = initial_state
        for _ in range(steps):
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
        self.step(initial_state=self.env.reset(), policy=self.random_policy, steps=transitions)

    def agent_select_action(self, state):
        return self.agent.select_action(np.array(state))

    def one_step(self):
        next_state, reward, done = self.step(initial_state=self.state, policy=self.agent_select_action, steps=1)
        self.state = next_state
        return reward, done


class TD3Runner(Runner):
    def __init__(self, env, agent, replay_buffer, max_eps_steps=200):
        super().__init__(env, agent, replay_buffer)
        self.max_eps_steps = max_eps_steps
        self.eps_steps = 0

    def step(self, initial_state, policy, steps=200):
        state = initial_state
        for step in range(steps):
            action = policy(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, next_state, reward, done)

            self.eps_steps = (self.eps_steps + 1) % self.max_eps_steps
            eps_finished = not self.eps_steps
            state = next_state
            if done or eps_finished:
                state = self.env.reset()

        return next_state, reward, done, eps_finished

    def one_step(self):
        next_state, reward, done, eps_finished = self.step(initial_state=self.state,
                                                           policy=self.agent_select_action,
                                                           steps=1)
        self.state = next_state
        return reward, done, eps_finished

    def train(self, batch_size=100, steps=int(1e6), eval_after=1000, avg_over=100):
        eps_num, eps_reward, eps_step = 0, 0, 0
        self.env.reset()
        rewards = []

        for step in range(steps):
            reward, done, eps_finished = self.one_step()
            eps_step += 1
            eps_reward += reward
            if done or eps_finished:

                for ep in range(eps_step):
                    state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
                    self.agent.train(state, action, next_state, reward.view(-1, 1), (1 - done).view(-1, 1), ep)

                rewards.append(eps_reward)
                avg_reward = np.mean(rewards[-avg_over:])

                eps_num += 1
                if (step + 1) % eval_after == 0:
                    evaluate_policy(self.agent, self.env)
                    self.env.reset()
                print(f'\rSteps: {step + 1} Episodes: {eps_num} Reward: {eps_reward:.2f} Avg Reward: {avg_reward:.2f}')
                eps_step, eps_reward = 0, 0

