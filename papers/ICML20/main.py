import argparse
import os
import copy
import time
import pickle

import numpy as np
import torch
import gym
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp
import pybullet_envs

from utils import env_params
from envs.pendulum import PendulumEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize bias for linear region
def linear_init_bias(bias_data, linear_radius=0.2):
    cb = 1.5
    m = linear_radius
    u = m*cb
    bias_data.uniform_(-u, u)

    while len(bias_data[abs(bias_data)<linear_radius]):
        bias_data[abs(bias_data)<linear_radius] = bias_data[abs(bias_data)<linear_radius].uniform_(-u, u)
    return bias_data


# Initialize effective bias for linear region
def linear_init_bias_l2(W2, b1, b2, linear_radius=0.2):
    c = 0.005
    m = linear_radius
    W2 = W2.clone()
    b1 = b1.clone().clamp_(0)

    while len((W2 @ b1 + b2)[abs(W2 @ b1 + b2)<m]):
        b2[abs(W2 @ b1 + b2)<m] += torch.sign(b2[abs(W2 @ b1 + b2)<m])*c
    return b2


# Post-training Fitting to Linear Controller
def opt_actor(l1, l2, l3, max_a, K, A=None, B=None, strict=True):
    W1 = l1.weight.data.clone().cpu().numpy()
    W2 = l2.weight.data.clone().cpu().numpy()
    W3 = l3.weight.data.clone().cpu().numpy()

    b1 = l1.bias.data.clone().unsqueeze(-1).cpu().numpy()
    b2 = l2.bias.data.clone().unsqueeze(-1).cpu().numpy()
    b3 = l3.bias.data.clone().unsqueeze(-1).cpu().numpy()

    W1p = W1.copy()
    b1p = b1.copy()
    idx1 = (b1p<0).nonzero()
    idx1 = (idx1[0], np.vstack((idx1[1], idx1[1]+1)))
    b1p[b1p<0] = 0
    W1p[idx1] = 0

    W2W1p = W2 @ W1p
    W2b1pb2 = W2 @ b1p + b2

    W2W1p_p = W2W1p.copy()
    W2b1pb2_p = W2b1pb2.copy()
    idx2 = (W2b1pb2_p<0).nonzero()
    idx2 = (idx2[0], np.vstack((idx2[1]+i for i in range(W2W1p_p.shape[1]))))
    W2b1pb2_p[W2b1pb2_p<0] = 0
    W2W1p_p[idx2] = 0

    K = torch.Tensor(K).numpy()

    Wopt = cp.Variable(W3.shape)
    bopt = cp.Variable(b3.shape)

    iterations = 0

    if strict:
        cost = 0
        constraints = []
        
        constraints.append(max_a*(Wopt @ W2W1p_p) + K == 0)
        constraints.append(Wopt @ (W2b1pb2_p) + bopt == 0)

        cost += cp.norm(Wopt - W3)
        cost += 0.5*cp.norm(bopt - b3)
        
        obj = cp.Minimize(cost)
        prob = cp.Problem(obj, constraints)
        prob.solve()
    else:
        eps = 0.0001
        nu_k = 1
        done = False

        while not done:
            iterations += 1
            cost = 0
            constraints = []

            constraints.append(cp.abs(Wopt @ (W2b1pb2_p) + bopt) <= eps)

            cost += cp.norm(Wopt - W3)
            cost += nu_k*cp.norm(max_a*(Wopt @ W2W1p_p) + K)
    
            obj = cp.Minimize(cost)
            prob = cp.Problem(obj, constraints)
            prob.solve()

            done = np.all(np.linalg.eig(A + B @ (max_a*Wopt.value @ W2W1p_p + Wopt.value @ (W2b1pb2_p) + bopt.value))[0] < 0)
            nu_k *= nu_k * 1.1

    # print('Optimization complete for Actor.')
    # print(f'CVX Problem Status: {prob.status}')
    # print(f'CVX Objective Value: {prob.value}')
    # print(f'--------------------------------------------')
    # print(f"Last layer diff: {np.linalg.norm(Wopt.value - W3)}")
    # print(f"LQR Fit: {np.linalg.norm(max_a*Wopt.value@W2W1p_p + K) + np.linalg.norm(Wopt.value @ (W2b1pb2_p) + bopt.value)}")
    # print(f"Max Closed Loop Eval: {np.max(np.linalg.eig(A + B @ (max_a*Wopt.value @ W2W1p_p + Wopt.value @ (W2b1pb2_p) + bopt.value))[0])}")
    # print(f'--------------------------------------------')
    l3.weight.data = torch.from_numpy(Wopt.value).to(device).float()
    l3.bias.data = torch.from_numpy(bopt.value).squeeze(-1).to(device).float()

    return None


# Bias-Shifted Architecture
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, lin_r=0.2, training=True, dropout_p=0.1):
        super(Actor, self).__init__()

        self.training = training
        self.lin_r = torch.Tensor([lin_r]).to(device)  # linear region
        self.dropout_p = dropout_p

        self.l1 = nn.Linear(state_dim, 512)
        linear_init_bias(self.l1.bias.data, linear_radius=lin_r)
        
        self.l2 = nn.Linear(512, 256)
        linear_init_bias(self.l2.bias.data, linear_radius=lin_r)
        linear_init_bias_l2(self.l2.weight.data, self.l1.bias.data, self.l2.bias.data, linear_radius=lin_r)
        
        self.l3 = nn.Linear(256, action_dim)
        nn.init.zeros_(self.l3.bias.data)  # linear region is zero for tanh

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(nn.functional.dropout(self.l1(state), p=self.dropout_p, training=self.training))
        a = F.relu(nn.functional.dropout(self.l2(a), p=self.dropout_p, training=self.training))
        a = nn.functional.dropout(self.l3(a), p=self.dropout_p, training=self.training)
        a = torch.tanh(a)
        return self.max_action * a


# Critic Architectures
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lin_r=0.2):
        super(Critic, self).__init__()
        self.lin_r = torch.Tensor([lin_r]).to(device)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        linear_init_bias(self.l1.bias.data, linear_radius=lin_r)

        self.l2 = nn.Linear(512, 256)
        linear_init_bias(self.l2.bias.data, linear_radius=lin_r)

        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 512)
        linear_init_bias(self.l4.bias.data, linear_radius=lin_r)

        self.l5 = nn.Linear(512, 256)
        linear_init_bias(self.l5.bias.data, linear_radius=lin_r)

        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = self.l2(q1)*torch.sigmoid(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = self.l5(q2)*torch.sigmoid(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = self.l2(q1)*torch.sigmoid(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        expl_noise=0.1,
        lin_r=0.2,
        training=True,
        dropout_p=0.1,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lin_r = lin_r
        self.training = training
        self.dropout_p = dropout_p

        self.actor = Actor(state_dim, action_dim, max_action, lin_r, training, dropout_p).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, lin_r).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.expl_noise = expl_noise

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def compute_actor_loss(self, state):
        return -self.critic.Q1(state, self.actor(state)).mean()

    def compute_critic_loss(self, target_Q, current_Q1, current_Q2):
        return F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

    def restrict_bias(self):
        self.actor.l1.bias.data[torch.abs(self.actor.l1.bias.data)<self.actor.lin_r] = torch.sign(self.actor.l1.bias.data[torch.abs(self.actor.l1.bias.data)<self.actor.lin_r])*self.actor.lin_r
        self.actor.l2.bias.data[torch.abs(self.actor.l2.bias.data)<self.actor.lin_r] = torch.sign(self.actor.l2.bias.data[torch.abs(self.actor.l2.bias.data)<self.actor.lin_r])*self.actor.lin_r

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = self.compute_critic_loss(target_Q, current_Q1, current_Q2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            actor_loss = self.compute_actor_loss(state)
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.restrict_bias()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        parameters = {
            "lin_r" : self.lin_r,
            "lamb" : self.lamb,
            "reg_type" : self.reg_type,
            "bias_reg" : self.bias_reg,
            "sig" : self.sig,
            "num_lin_r" : self.num_lin_r,
            "K" :self.K,
            "lamb_last_bias" :self.lamb_last_bias,
            "lamb_fit_k" :self.lamb_fit_k,
            "lamb_2nd_bias" :self.lamb_2nd_bias,
            "dropout_p" :self.dropout_p,
        }
        torch.save(parameters, filename + "_parameters")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        if os.path.exists(filename + "_parameters"):
            parameters = torch.load(filename + "_parameters")
            self.lin_r = parameters["lin_r"]
            self.lamb = parameters["lamb"]
            self.reg_type = parameters["reg_type"]
            self.bias_reg = parameters["bias_reg"]
            self.sig = parameters["sig"]
            self.num_lin_ = parameters["num_lin_r"]
            self.actor.lin_r = torch.Tensor(np.array(parameters["lin_r"])).to(device)
            self.critic.lin_r = torch.Tensor(np.array(parameters["lin_r"])).to(device)
            if "K" in parameters.keys():
                self.K = parameters["K"]
                self.lamb_last_bias = parameters["lamb_last_bias"]
                self.lamb_fit_k = parameters["lamb_fit_k"]
                self.lamb_2nd_bias = parameters["lamb_2nd_bias"]
                self.dropout_p = parameters["dropout_p"]
                self.actor.dropout_p = parameters["dropout_p"]


class TD3ExplicitLoss(TD3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lamb = 0.001
        self.reg_type = "scaled_sign"
        self.bias_reg = True
        self.sig = 2.
        self.num_lin_r = 1.2
        self.K = [[3.58891436, 1.86675319]]  # Input desired K to fit to here
        self.lamb_1st_bias = 100.0
        self.lamb_2nd_bias = 6.0
        self.lamb_last_bias = 1.2
        self.lamb_fit_k = 2.5

    def compute_actor_loss(self, state):
        l2_reg = torch.Tensor([0.0]).to(device)

        W1p = torch.where(self.actor.l1.bias.unsqueeze(-1).view(-1,1).repeat(1,self.state_dim) < 0, torch.zeros_like(self.actor.l1.weight), self.actor.l1.weight)
        W2W1p = self.actor.l2.weight @ W1p
        
        b1p = self.actor.l1.bias.data.clone()
        b1p[b1p<0] = 0
        W2b1pb2 = self.actor.l2.weight @ b1p + self.actor.l2.bias
        W2W1p_p = torch.where(W2b1pb2.data.unsqueeze(-1).view(-1,1).repeat(1,self.state_dim) < 0, torch.zeros_like(W2W1p), W2W1p)

        W2b1pb2_p = W2b1pb2.data.clone()
        W2b1pb2_p[W2b1pb2_p < 0] = 0

        W1z = torch.sign(self.actor.l1.weight.data.clone())*.707
        W2W1z = torch.sign(W2W1p.clone())*.707

        if self.reg_type == "scaled_sign":
            l2_reg += 0.2*torch.norm(self.actor.l1.weight - W1z)
            l2_reg += 0.2*torch.norm(W2W1p - W2W1z)

        # Stay close to K from LQR in linear region
        KT = torch.Tensor(self.K).to(device)
        if self.action_dim == 1:
            l2_reg += self.lamb_fit_k*torch.norm(self.actor.max_action*(self.actor.l3.weight @ W2W1p_p) + KT)

        # Reduce effective bias as last layer
        l2_reg += self.lamb_last_bias*torch.norm(self.actor.l3.weight @ W2b1pb2_p + self.actor.l3.bias)

        # Added for bias term make this small and b2 relatively large
        l2_reg += self.lamb_2nd_bias*torch.norm(self.actor.l2.weight @ b1p)

        return super().compute_actor_loss(state) + self.lamb*l2_reg.squeeze(-1)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


def eval_policy(policy, eval_env, seed, eval_episodes=10):
    eval_env.seed(seed + 100)

    avg_reward = 0.
    rewards = []
    for _ in range(eval_episodes):
        episode_reward = 0.0
        state, done = eval_env.reset(), False
        count = 0
        while not done and count < 1000:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            episode_reward += reward
            count += 1
        rewards.append(episode_reward)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return rewards


class TD3Trainer(object):
    def __init__(self,
                 env, 
                 replay_buffer,
                 policy,
                 eval_policy,
                 ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.eval_policy = eval_policy
        self.total_steps_trained = 0

    def populate_replay_buffer(self,
                               replay_samples=1e4
                               ):

        print(f"Populating replay buffer with {int(replay_samples)} samples")
        state = self.env.reset()

        self.total_steps_trained += int(replay_samples)
        for t in range(int(replay_samples)):
            action = self.env.action_space.sample()

            # Perform action
            next_state, reward, done, _ = self.env.step(action) 

            # Store data in replay buffer
            self.replay_buffer.add(state, action, next_state, reward, float(done))

            state = next_state

    def train(self,
              train_steps=1e6, 
              batch_size=256,
              eval_freq=5000,
              avg_freq=1000,
              max_steps_before_reset=1000
              ):

        state = self.env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        # Evaluate untrained policy
        evaluations = {}
        self.policy.training = False
        evaluations[self.total_steps_trained] = self.eval_policy(self.policy)
        self.policy.training = True

        for t in range(int(train_steps)):
            self.total_steps_trained += 1
            episode_timesteps += 1

            action = (
                self.policy.select_action(np.array(state))
                + np.random.normal(0, self.policy.max_action * self.policy.expl_noise, size=self.policy.action_dim)
            ).clip(-self.policy.max_action, self.policy.max_action)

            # Perform action
            next_state, reward, done, _ = self.env.step(action) 
            done_bool = float(done) if episode_timesteps < self.env._max_episode_steps else 0

            # Store data in replay buffer
            self.replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            self.policy.train(self.replay_buffer, batch_size)

            if (self.total_steps_trained) % 1000 == 0:
                print(f'Total Steps: {self.total_steps_trained}')
            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total Steps: {self.total_steps_trained} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state, done = self.env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            if episode_timesteps >= max_steps_before_reset:
                print(f"Total Steps: {self.total_steps_trained} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} [Reset]")
                state, done = self.env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % eval_freq == 0:
                self.policy.training = False
                evaluations[str(self.total_steps_trained)] = self.eval_policy(self.policy)
                self.policy.training = True

        return evaluations

    def save(self, filename):
        self.policy.save(filename)
        torch.save(self.replay_buffer, filename + "_replay_buffer")
        torch.save(self.total_steps_trained, filename + "_total_steps_trained")

    def load(self, filename):
        self.policy.load(filename)
        self.replay_buffer = torch.load(filename + "_replay_buffer")
        self.total_steps_trained = torch.load(filename + "_total_steps_trained")

def setup_parameters(args,
                     kwargs,
                     TD3,
                     ReplayBuffer,
                     make_env
                     ):

    env = make_env()

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    dims = env_params(env)
    s_dim, a_dim, max_a, min_a = dims

    state_dim = s_dim
    action_dim = a_dim 
    max_action = float(max_a)

    kwargs["state_dim"] = state_dim
    kwargs["action_dim"] = action_dim
    kwargs["max_action"] = max_action
    kwargs["discount"] = args.discount
    kwargs["tau"] = args.tau

    # Initialize policy: target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    kwargs["expl_noise"] = args.expl_noise
    policy = TD3(**kwargs)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    eval_env = make_env()
    eval_env.seed(args.seed+100)
    eval_env._max_episode_steps = 1000

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    eval_policy_func = lambda policy_a : eval_policy(policy_a, eval_env, args.seed)

    ################### REAL ENV STUFF ######################
    USE_REAL_ENV = False

    if USE_REAL_ENV:
        pass

    ################# END REAL ENV STUFF ####################

    LOAD_FILE_PATH = args.load_file_path

    env._max_episode_steps = 1000
    trainer = TD3Trainer(env, replay_buffer, policy, eval_policy_func)
    if LOAD_FILE_PATH is not None:
        trainer.load(LOAD_FILE_PATH)
    policy = trainer.policy

    return policy, trainer, eval_policy_func

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3BShift")
    
    # TD3BShift --Full implementation of our architecture initialization and regularization terms
    # TD3 --original implementation of TD3
    # TD3Mod --TD3 with bias-shifted Actor and our Critic architecture
    # TD3Restrict --TD3Mod with restricted updates on bias so it maintains a magnitude
    # TD3Loss --TD3 with our regularization terms added to the loss

    parser.add_argument("--env", default="PendulumEnv")
    ### GYM PYBULLET ENVS ###
    # InvertedPendulumBulletEnv-v0
    # InvertedDoublePendulumBulletEnv-v0
    # ReacherBulletEnv-v0
    # HopperBulletEnv-v0
    # HalfCheetahBulletEnv-v0
    # AntBulletEnv-v0

    ### CUSTOM ###
    # PendulumEnv

    parser.add_argument("--seed", default=0, type=int)              # Set seed
    parser.add_argument("--lin_r", default=0.4)                     # Size of linear region
    parser.add_argument("--start_timesteps", default=1e4, type=int) # Random policy used to collect transitions
    parser.add_argument("--eval_freq", default=5e3, type=int)       # Evaluation frequency
    parser.add_argument("--train_steps", default=1e5, type=int)     # Number of transitions trained on
    parser.add_argument("--expl_noise", default=0.1)                # Standard deviation used for action exploration
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for actor and critic
    parser.add_argument("--discount", default=0.99)                 # Gamma, discount factor
    parser.add_argument("--tau", default=0.005)                     # Polyak Averaging target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update for smoothing
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of policy updates
    parser.add_argument("--dropout", default=0.2)                   # Dropout rate to create redundant features
    parser.add_argument("--load_model", default=None, type=str)     # Relative file path to load
    
    parser.add_argument("--optimize_lastlayer", action="store_true")# ONLY USE W/ PendulumEnv unless have known controller
    args = parser.parse_args()

    # Load Policy Type
    if args.policy == "TD3BShift":
        TD3Type = TD3ExplicitLoss
        run_type = "bias_shifted"
    elif args.policy == "TD3":
        from td3.TD3 import TD3
        TD3Type = TD3
        run_type = "modified"
    elif args.policy == "TD3Mod":
        from td3.TD3Mod import TD3Mod
        TD3Type = TD3Mod
        run_type = "modified"
    elif args.policy == "TD3Restrict":
        from td3.TD3Restrict import TD3Restrict
        TD3Type = TD3Restrict
        run_type = "modified"
    elif args.policy == "TD3Loss":
        from td3.TD3Loss import TD3Loss
        TD3Type = TD3Loss
        run_type = "modified"
    
    # Load Environment
    if args.env == "PendulumEnv":
        th_init = 180*np.pi/180
        thdot_init = 0.1
        max_speed = 100.0

        max_torque = 0.8
        exit_reward = 1000
        b = 0.1
        dt = 0.2

        env_kwargs = {
        "max_speed": max_speed,
        "max_torque": max_torque,
        "b" : b,
        "th_init": (-th_init, th_init), 
        "thdot_init": (-thdot_init, thdot_init), 
        "step_dt": dt, 
        "exit_reward": exit_reward,
        }  

        make_env = lambda : PendulumEnv(**env_kwargs)
    
    else:
        make_env = lambda: gym.make(args.env).env

    eval_env = make_env()
    eval_env.seed(args.seed+100)
    eval_env._max_episode_steps = 1000

    train_steps = args.train_steps
    reset_max_steps = 1500  # Reset environment after steps, added for convergence for TD3

    args.load_file_path = args.load_model

    kwargs = {"lin_r": args.lin_r,
              "dropout_p" : args.dropout}
    
    # Make policy
    policy, trainer, _ = setup_parameters(args, kwargs, TD3Type, ReplayBuffer, make_env)
 
    # Get stabilizing K from environment
    if args.env == "PendulumEnv":
        K, _ = eval_env.lqr_kp()

    # Save evaluations
    folder = "saves/evals/"
    os.makedirs(folder, exist_ok=True)
    ext = ".pkl"
    fn = folder + run_type + ext
    os.makedirs(f"saves/{run_type}", exist_ok=True)

    # Train
    trainer.save(f"saves/{run_type}/{run_type}")
    trainer.populate_replay_buffer(replay_samples=1e4)
    evals = {}
    for k in range(1):
        evalsk = trainer.train(train_steps=train_steps, max_steps_before_reset=reset_max_steps, eval_freq=5000)
        evals.update(evalsk)
        trainer.save(f"saves/{run_type}/{run_type}{k}")

    with open(fn, "wb") as f:
        pickle.dump(evals, f)
    policy = trainer.policy

    # Optimize last layer weights
    if args.env == "PendulumEnv" and args.optimize_lastlayer:
        policy.actor.training = False
        A, B, _, _ = eval_env.get_state_matrices()
        opt_actor(policy.actor.l1, policy.actor.l2, policy.actor.l3, max_torque, K, A=A, B=B, strict=True)

    # Visualize Policy
    max_steps = 400
    eval_env.render(mode='human')
    while True:
        state = eval_env.reset()
        step = 0
        done = False
        while (not done) and (step < max_steps): 
            state, _, done, _ = eval_env.step(policy.select_action(state))
            step += 1
            print('Done!') if done else 0
        print(f'Resetting from step {step}/{max_steps}. Starting new run...')
