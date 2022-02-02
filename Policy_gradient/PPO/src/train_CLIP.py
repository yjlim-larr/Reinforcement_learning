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

ACTOR_PT = "actor_CLIP.pt"
CRITIC_PT = "critic_CLIP.pt"

ACTOR_PATH = "./" + ACTOR_PT
CRITIC_PATH = "./" + CRITIC_PT

# environment
env = gym.make('Pendulum-v1')


# dimension
ob = env.reset()
state_dim = ob.size
action_max = env.action_space.high.item() # continuous action
action_min = env.action_space.low.item()


# actor, critic
actor = model.Actor(state_dim, action_min, action_max, param.hidden)
critic = model.Critic(state_dim, param.hidden)


# load weight
if os.path.isfile(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH))
else:
    actor.apply(utils.init_weight)

if os.path.isfile(CRITIC_PATH):
    critic.load_state_dict(torch.load(CRITIC_PATH))
else:
    critic.apply(utils.init_weight)


# Memory
mem = utils.Memory(param.capacity)


# train
critic_optim = optim.Adam(critic.parameters(), lr = 0.0003)
actor_optim = optim.Adam(actor.parameters(), lr = 0.0003)

for k in range(param.episodes):
    states = []
    next_states = []
    actions = []
    action_probs = []
    returns = []
    rewards = []
    dones = []

    prev_state = env.reset()
    prev_state = torch.tensor(prev_state, dtype=torch.float, requires_grad=True).view(1, -1)

    while True:
        while True:
            states.append(prev_state)

            # get action
            mu, std = actor(prev_state)
            m = torch.distributions.normal.Normal(loc=mu, scale=std)
            action = m.sample().view(1, -1)

            prob = torch.exp(m.log_prob(action).view(1, -1))

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


        # GAE
        values, ads = utils.GAE(critic, rewards, dones, states, next_states, param.gamma, param.lam)

        # Memory push
        for i in range(len(rewards)):
            mem.push(states[i], next_states[i], actions[i], action_probs[i], values[i], ads[i], rewards[i], dones[i])

            if mem.get_size() >= param.capacity:
                break

        # Full Memory
        if mem.get_size() >= param.capacity:
            break


    mem.shuffle()
    size = mem.get_size()
    for i in range(size // param.batch_size):
        print("Episodes: ", k, "epochs: ", i)
        states, next_states, actions, action_probs, values, ads, rewards, dones = mem.sample(i * param.batch_size, param.batch_size)

        # train
        utils.CLIP_train(actor, actor_optim, states, actions, action_probs, ads, param.ppo_epsilon, critic, critic_optim, values)

        print("___________________________________________________________________________________________")

    # Memory reset
    mem.clear()

    # Save
    torch.save(actor.state_dict(), ACTOR_PT)
    torch.save(critic.state_dict(), CRITIC_PT)

# finish env
env.close()