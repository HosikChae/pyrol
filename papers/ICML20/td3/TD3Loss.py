import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Loss(object):
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

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.expl_noise = 0.1

        self.total_it = 0

        self.lamb = 0.001
        self.reg_type = "simp" # "sval", "sing", "simp"
        self.bias_reg = True
        self.sig = 2.
        self.num_lin_r = 1.2
        # self.K = [[33.12046239, 10.08024327]]
        self.K = [[3.58891436, 1.86675319]]
        self.lamb_1st_bias = 100.0
        self.lamb_2nd_bias = 4.0
        self.lamb_last_bias = 20.0
        self.lamb_fit_k = 20.0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


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
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = self.compute_actor_loss(state)
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def compute_actor_loss(self, state):
        sval256 = torch.eye(256)[0:2].T.to(device)
        sval512 = torch.eye(512)[0:2].T.to(device)

        l2_reg = torch.Tensor([0.0]).to(device)

        b1_reg = self.actor.l1.bias.data.clone()

        W1p = torch.where(self.actor.l1.bias.unsqueeze(-1).view(-1,1).repeat(1,self.state_dim) < 0, torch.zeros_like(self.actor.l1.weight), self.actor.l1.weight)
        W2W1p = self.actor.l2.weight @ W1p
        
        b1p = self.actor.l1.bias.data.clone()
        b1p[b1p<0] = 0
        W2b1pb2 = self.actor.l2.weight @ b1p + self.actor.l2.bias
        W2W1p_p = torch.where(W2b1pb2.data.unsqueeze(-1).view(-1,1).repeat(1,self.state_dim) < 0, torch.zeros_like(W2W1p), W2W1p)
        
        W2b1pb2_reg = W2b1pb2.data.clone()

        W2b1pb2_p = W2b1pb2.data.clone()
        W2b1pb2_p[W2b1pb2_p < 0] = 0

        W1z = torch.sign(self.actor.l1.weight.data.clone())*.707
        W2W1z = torch.sign(W2W1p.clone())*.707

        if self.reg_type == "sval":
            l2_reg += torch.norm(self.actor.l1.weight - sval512)
            l2_reg += torch.norm(W2W1p - sval256)
            l2_reg += torch.norm(self.actor.l3.weight @ W2W1p_p - sval512)
        elif self.reg_type == "sing":
            l2_reg += 0.1*torch.norm(torch.norm(W1p, 2) - self.sig)
            l2_reg += 0.1*torch.norm(torch.norm(W2W1p, 2) - self.sig)
            l2_reg += 0.2*torch.norm(self.actor.l3.weight @ W2W1p_p)

        elif self.reg_type == "simp":
            l2_reg += 0.2*torch.norm(self.actor.l1.weight - W1z)
            l2_reg += 0.2*torch.norm(W2W1p - W2W1z)

            l2_reg += torch.norm(self.actor.l3.weight @ W2W1p_p)


        KT = torch.Tensor(self.K).to(device)

        l2_reg += self.lamb_2nd_bias*torch.norm(self.actor.l2.weight @ b1p)


        return -self.critic.Q1(state, self.actor(state)).mean() + self.lamb*l2_reg.squeeze(-1)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
