import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import model
import utils
import param

import os
from itertools import count
ACTOR_PATH = "./actor.pt"
CRITIC_PATH = "./critic.pt"

env = gym.make('CartPole-v0')
env.reset()

# dimension
ob = env.reset()
state_dim = ob.size
action_dim = env.action_space.n

# actor, critic
actor = model.Actor(state_dim, action_dim)
critic = model.Critic(state_dim, action_dim)

# load weight
if os.path.isfile(ACTOR_PATH):
    actor.load_state_dict(torch.load(ACTOR_PATH))
else:
    actor.apply(utils.init_weight)

if os.path.isfile(CRITIC_PATH):
    critic.load_state_dict(torch.load(CRITIC_PATH))
else:
    critic.apply(utils.init_weight)

state = env.reset()
for i in count():
    env.render()

    state = torch.tensor(state, dtype=torch.float, requires_grad=True).view(1, -1)
    act = actor(state)
    action = Categorical(act).sample()
    next_state, reward, done, info = env.step(action.item())
    state = next_state

    if done:
        break

env.close()