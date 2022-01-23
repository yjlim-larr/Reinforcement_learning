import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import param
import model
import utils

import gym
import os
import random

ACTOR_PATH = "./actor.pt"
CRITIC_PATH = "./critic.pt"

# environment
env = gym.make('Pendulum-v1')


# dimension
ob = env.reset()
state_dim = ob.size
action_max = env.action_space.high.item() # continuous action
action_min = env.action_space.low.item()


# actor, critic
hidden = [100, 50, 25]
actor = model.Actor(state_dim, action_min, action_max, hidden)
critic = model.Critic(state_dim, hidden)


# load weight
if os.path.isfile(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH))
else:
    actor.apply(utils.init_weight)

if os.path.isfile(CRITIC_PATH):
    critic.load_state_dict(torch.load(CRITIC_PATH))
else:
    critic.apply(utils.init_weight)


# train
for k in range(param.episodes):
    print(k)

    states = []
    next_states = []
    actions = []
    action_probs = []
    returns = []
    rewards = []
    dones = []

    while True:
        prev_state = env.reset()
        prev_state = torch.tensor(prev_state, dtype=torch.float, requires_grad=True).view(1, -1)

        for j in range(200):
            states.append(prev_state)

            # get action
            mu, std = actor(prev_state)
            m = torch.distributions.normal.Normal(loc=mu, scale=std)
            action = m.sample().view(1, -1)

            prob = m.log_prob(action).view(1, -1)

            next_state, reward, done, _ = env.step(action.detach().numpy())
            next_state = torch.tensor(next_state, dtype=torch.float, requires_grad=True).view(1, -1)

            next_states.append(next_state.view(1,-1))

            actions.append(action.view(1,-1))
            action_probs.append(prob.view(1,-1))

            rewards.append(torch.tensor(reward).view(1, -1))
            dones.append(torch.tensor(int(done)).view(1, -1))

            if done == 1:
                break

            prev_state = next_state


        dones[-1] = torch.tensor(1).view(1, -1)
        if(len(states)) > param.capacity:
            break

    # make trajectory tensor data
    states = torch.cat(states, dim=0)
    next_states = torch.cat(next_states, dim=0)

    actions = torch.cat(actions, dim=0)
    action_probs = torch.cat(action_probs, dim=0)

    rewards = torch.cat(rewards, dim=0)

    dones = torch.cat(dones, dim = 0)


    # GAE
    returns = utils.GAE(critic, rewards, dones, states, next_states, param.gamma, param.lam)


    # train_actor
    utils.train_actor(actor, critic, states, actions, action_probs, returns, param.max_kl, param.tol)


    # train_critic
    utils.train_critic2(critic, states, next_states, rewards, dones, param.gamma, param.epsilon)


    print("___________________________________________________________________________________________")
    torch.save(actor.state_dict(), 'actor.pt')
    torch.save(critic.state_dict(), 'critic.pt')

env.close()
