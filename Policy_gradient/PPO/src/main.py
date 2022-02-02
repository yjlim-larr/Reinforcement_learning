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


with torch.no_grad():
    prev_state = env.reset()

    while True:
        env.render()
        # get action
        prev_state = torch.tensor(prev_state).view(1,-1)
        mu, std = actor(prev_state)
        m = torch.distributions.normal.Normal(loc=mu, scale=std)

        action = m.sample().view(1, -1)
        print("actions: ", torch.exp(m.log_prob(action)))

        next_state, reward, done, _ = env.step(action.detach().numpy())
        print("reward: ", reward)

        if done == 1:
            break

        prev_state = next_state

env.close()