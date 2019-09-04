import gym
import numpy as np
import torch
import torch.nn.functional as F

from pyrol.approximators.nn import Actor, TD3Critic
from pyrol.buffers import TorchReplayBasic
from pyrol.runners import TD3Runner
from pyrol.utils.maths import polyak_avg
from pyrol.utils.pytorch import np2torch_float


class TD3(object):
    def __init__(self, env, tau=0.005, gamma=0.99, update_freq=2, device='cpu'):
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

        # Scale noise by env action magnitude
        self.select_action_noise = self.max_a / 10
        self.policy_noise = self.select_action_noise * 2
        self.noise_clip = self.policy_noise * 2

        self.update_freq = update_freq
        self.device = device

    def select_action(self, state, noise=True):
        state = torch.from_numpy(state.reshape(1, -1)).float().to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise:
            action += np.random.normal(0, self.select_action_noise, size=self.a_dim)

        return action.clip(self.min_a, self.max_a)

    def train(self, state, action, next_state, reward, done, ep):
        # Noisy action
        noise = action.clone().normal_(0, self.policy_noise).to(self.device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
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
        if ep % self.update_freq == 0:

            # Actor loss
            actor_loss = -self.critic.q1(torch.cat([state, self.actor(state)], 1)).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(polyak_avg(target_param.data, param.data, self.tau))

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(polyak_avg(target_param.data, param.data, self.tau))

    def save(self, filename, directory):
        # TODO: Implement Save and Load functions
        # torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        # torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        pass

    def load(self, filename, directory):
        # self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        # self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        pass


if __name__ == '__main__':
    SEED = 0

    from pyrol.envs import gym_register_envs
    gym_register_envs(sim='maths')
    env = gym.make('PendulumMaths-v0').env  # "PendulumMaths-v0", 'Pendulum-v0'  TODO: unwrap time wrapper
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    agent = TD3(env, device=device)
    replay_buffer = TorchReplayBasic(device=device, conversion=np2torch_float, capacity=1000000)
    runner = TD3Runner(env, agent, replay_buffer, max_eps_steps=400)

    runner.populate_replay_buffer()
    runner.train()

