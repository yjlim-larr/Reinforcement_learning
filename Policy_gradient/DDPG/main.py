import model
import param
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

import os
import random
import matplotlib.pyplot as plt

ACTOR_PATH = "./actor.pt"
CRITIC_PATH = "./critic.pt"

env = gym.make("Pendulum-v1")
# dimension
ob = env.reset()
state_dim = ob.size
action_max = env.action_space.high.item()
action_min = env.action_space.low.item()

# replay mem
replay_mem = utils.Replay_mem(param.capacity)

# Teacher
actor = model.Actor(state_dim, action_high = action_max, action_low = action_min)
critic = model.Critic(state_dim, 1)

# Target
target_actor = model.Actor(state_dim, action_high = action_max, action_low = action_min)
target_critic = model.Critic(state_dim, 1)

# load weight
if os.path.isfile(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH))
else:
    actor.apply(utils.init_weights)

if os.path.isfile(CRITIC_PATH):
    critic.load_state_dict(torch.load(CRITIC_PATH))
else:
    critic.apply(utils.init_weights)

target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())


with torch.no_grad():
    prev_state = env.reset()

    for i in range(200):
        env.render()
        prev_state = torch.tensor(prev_state, dtype=torch.float).view(1, -1)
        action = target_actor(prev_state)

        next_state, reward, done, _ = env.step(action.detach().numpy())
        prev_state = next_state
env.close()
