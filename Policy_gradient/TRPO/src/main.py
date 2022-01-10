import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import param
import model
import utils

import gym
import os
from collections import deque

ACTOR_PATH = "./actor.pt"
CRITIC_PATH = "./critic.pt"

# environment
env = gym.make('Pendulum-v1')


# dimension
ob = env.reset()
state_dim = ob.size
action_max = env.action_space.high.item()
action_min = env.action_space.low.item()


# actor, critic
actor = model.Actor(state_dim, action_min, action_max)
critic = model.Critic(state_dim)


# load weight
if os.path.isfile(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH))
else:
    actor.apply(utils.init_weight)

if os.path.isfile(CRITIC_PATH):
    critic.load_state_dict(torch.load(CRITIC_PATH))
else:
    critic.apply(utils.init_weight)

    
    
with torch.no_grad():
    prev_state = env.reset()

    while True:
        env.render()
        
        mu, std = actor(torch.tensor(prev_state).view(1,-1))
        print(mu, std)
        m = torch.distributions.normal.Normal(loc=mu, scale=std)

        action = m.sample().view(1, -1).detach().numpy()

        next_state, reward, done, _ = env.step(action)

        if done == 1:
            break

        prev_state = next_state

env.close()
