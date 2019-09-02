import gym
import numpy as np
import torch
import torch.nn.functional as F

from pyrol.approximators.nn import Actor, TD3Critic
from pyrol.buffers import ReplayBasic
from pyrol.runners import Runner
from pyrol.utils.math_utils import polyak_avg


class TD3(object):
    def __init__(self, env, tau=0.005, gamma=0.99, device='cpu'):
        self.env = env
        self.s_dim = np.prod(self.env.observation_space.shape)
        self.a_dim = np.prod(self.env.action_space.shape)
        self.max_a = env.action_space.high.item()
        self.min_a = env.action_space.low.item()

        self.actor = Actor(self.s_dim, self.a_dim, self.max_a).to(device)
        self.actor_target = Actor(self.s_dim, self.a_dim, self.max_a).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = TD3Critic(self.s_dim, self.a_dim).to(device)
        self.critic_target = TD3Critic(self.s_dim, self.a_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.tau = tau
        self.gamma = gamma
        self.device = device

    def select_action(self, state, noise=0.1):
        state = torch.from_numpy(state.reshape(1, -1)).float().to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten() + np.random.normal(0, noise, size=self.a_dim)

        return action.clip(self.min_a, self.max_a)

    def train(self,
              replay_buffer,
              updates,
              batch_size=100,
              policy_noise=0.2,
              noise_clip=0.5,
              policy_freq=2):

        for update in range(updates):

            state, action, next_state, reward, done = replay_buffer.sample(batch_size)

            state = torch.from_numpy(state).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            reward = torch.from_numpy(np.expand_dims(reward, axis=1)).float().to(self.device)
            done = torch.from_numpy(np.expand_dims((1-done), axis=1)).float().to(self.device)

            # Noisy action
            noise = action.clone().normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(-self.max_a, self.max_a)

            # Bellman update
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            min_q = torch.min(target_q1, target_q2)
            y = (reward + done * self.gamma * min_q).detach()

            # Critic loss
            q1, q2 = self.critic(state, action)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

            # Optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if update % policy_freq == 0:

                # Actor loss
                actor_loss = -self.critic.q1(torch.cat([state, self.actor(state)], 1)).mean()

                # Optimize actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(polyak_avg(target_param.data, param.data, self.tau))

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(polyak_avg(target_param.data, param.data, self.tau))

    def save(self, filename, directory):
        # TODO: Implement Save and Load functions
        # torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        # torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        pass

    def load(self, filename="best", directory="./log"):
        # self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        # self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        pass


def train(agent, env, runner, steps=int(5e6)):
    eps_num, eps_reward, eps_step = 0, 0, 0
    env.reset()
    rewards = []

    for step in range(steps):
        # TODO: Implement step until done
        # TODO: Get rid of false signals when horizon ends as in paper
        reward, done = runner.one_step()
        eps_step += 1
        eps_reward += reward
        if done:
            agent.train(replay_buffer, eps_step)
            rewards.append(eps_reward)
            avg_100_reward = np.mean(rewards[-100:])

            eps_num += 1
            print(f'\rSteps: {step} Episodes: {eps_num} Reward: {eps_reward:.2f} Avg Reward: {avg_100_reward:.2f}')
            eps_step, eps_reward = 0, 0


if __name__ == '__main__':
    SEED = 0

    # from pyrol.envs import gym_register_envs
    # gym_register_envs(sim='maths')
    env = gym.make('Pendulum-v0')  # "PendulumMaths-v0"  TODO: unwrap time wrapper
    # TODO: fix faulty done signal in replay buffer
    # TODO: add valid done signal in PendulumMaths-v0 file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    agent = TD3(env, device=device)
    replay_buffer = ReplayBasic(capacity=1000000)
    runner = Runner(env, agent, replay_buffer)

    runner.populate_replay_buffer()
    train(agent, env, runner)

