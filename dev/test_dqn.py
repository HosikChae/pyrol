import gym
import sys
import torch
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from pyrol.approximators.nn import Actor, TD3Critic
from pyrol.buffers import ReplayBasic


class TD3(object):

    def __init__(self, state_dim, action_dim, max_action, env):
        self.actor = Actor(state_dim, action_dim, max_action).to(device).double()
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device).double()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = TD3Critic(state_dim, action_dim).to(device).double()
        self.critic_target = TD3Critic(state_dim, action_dim).to(device).double()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.env = env

    def select_action(self, state, noise=0.1):
        state = torch.DoubleTensor(state.reshape(1, -1)).to(device)

        action = self.actor(state).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Sample replay buffer
            state, action, next_state, reward, done = replay_buffer.sample(batch_size)

            state = torch.from_numpy(state).double().to(device)
            action = torch.from_numpy(action).double().to(device)
            next_state = torch.from_numpy(next_state).double().to(device)
            reward = torch.from_numpy(np.expand_dims(reward, axis=1)).double().to(device)
            done = torch.from_numpy(np.expand_dims((1-done), axis=1)).double().to(device)


            # Select action according to policy and add clipped noise
            noise = action.clone().normal_(0, policy_noise).to(device)  # Need to detach
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q)  # .detach() not needed?

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.q1_out(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="./saves"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


class Runner(object):
    """Runs experiments and pushes experience to replay buffer"""

    def __init__(self, env, agent, replay_buffer):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done = False

    def next_step(self, episode_timesteps, noise=0.1):
        action = self.agent.select_action(np.array(self.obs), noise=noise)

        # Perform action
        new_obs, reward, done, _ = self.env.step(action)
        done_bool = 0 if episode_timesteps + 1 == 200 else float(done)

        replay_buffer.push(self.obs, action, new_obs, reward, done_bool)

        self.obs = new_obs

        if done:
            self.obs = self.env.reset()
            done = False

            return reward, True

        return reward, done


# Evaluate
def evaluate_policy(policy, env, eval_episodes=100, render=False):
    """run several episodes using the best agent policy

        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training

        Returns:
            avg_reward (float): average reward over the number of evaluations

    """

    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = policy.select_action(np.array(obs), noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


# Observation
def observe(env, replay_buffer, observation_steps):
    """run episodes while taking random actions and filling replay_buffer

        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for

    """

    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.push(obs, action, new_obs, reward, done)

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()


# Train
def train(agent, test_env):
    """Train the agent for exploration steps

        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            writer (SummaryWriter): tensorboard writer
            exploration (int): how many training steps to run

    """

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False
    obs = env.reset()
    evaluations = []
    rewards = []
    best_avg = -2000

    writer = SummaryWriter(comment="-TD3_Baseline_HalfCheetah")

    while total_timesteps < EXPLORATION:

        if done:

            if total_timesteps != 0:
                rewards.append(episode_reward)
                avg_reward = np.mean(rewards[-100:])

                writer.add_scalar("avg_reward", avg_reward, total_timesteps)
                writer.add_scalar("reward_step", reward, total_timesteps)
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)

                if best_avg < avg_reward:
                    best_avg = avg_reward
                    print("saving best model....\n")
                    agent.save("best_avg", "saves")

                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                    total_timesteps, episode_num, episode_reward, avg_reward), end="")
                sys.stdout.flush()

                if avg_reward >= REWARD_THRESH:
                    break

                agent.train(replay_buffer, episode_timesteps, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP,
                            POLICY_FREQUENCY)

                # Evaluate episode
                #                 if timesteps_since_eval >= EVAL_FREQUENCY:
                #                     timesteps_since_eval %= EVAL_FREQUENCY
                #                     eval_reward = evaluate_policy(agent, test_env)
                #                     evaluations.append(avg_reward)
                #                     writer.add_scalar("eval_reward", eval_reward, total_timesteps)

                #                     if best_avg < eval_reward:
                #                         best_avg = eval_reward
                #                         print("saving best model....\n")
                #                         agent.save("best_avg","saves")

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

        reward, done = runner.next_step(episode_timesteps)
        episode_reward += reward

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


ENV = 'Pendulum-v0'  # "PendulumMaths-v0"
SEED = 0
OBSERVATION = 10000
EXPLORATION = 5000000
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
EVAL_FREQUENCY = 5000
REWARD_THRESH = 8000

if __name__ == '__main__':
    # from pyrol.envs import gym_register_envs
    # gym_register_envs(sim='maths')
    env = gym.make(ENV)
    device = 'cpu'  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(state_dim, action_dim, max_action, env)
    replay_buffer = ReplayBasic(capacity=1000000)
    runner = Runner(env, policy, replay_buffer)

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    # Populate replay buffer
    observe(env, replay_buffer, OBSERVATION)

    # Train agent
    train(policy, env)

    policy.load()

    for i in range(100):
        evaluate_policy(policy, env, render=True)


# def plot_durations():
#     plt.figure(3)
#     plt.clf()
#     durations_t = torch.tensor(episode_rewards, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Rewards')
#     plt.plot(durations_t.numpy())
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())
#
#     plt.pause(0.001)
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())

