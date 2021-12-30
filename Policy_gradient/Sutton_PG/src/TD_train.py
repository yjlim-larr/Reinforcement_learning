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

# test game
for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if(done == 1):
        break
env.close()

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

# optimizer
actor_optim = optim.Adam(actor.parameters())
critic_optim = optim.Adam(critic.parameters())


actor_optim = optim.Adam(actor.parameters())
critic_optim = optim.Adam(critic.parameters())
for _ in range(param.epochs):
    state = env.reset()
    log_probs = []
    values = []
    rewards = []
    masks = []

    for i in count():
        #env.render()
        state = torch.tensor(state, dtype=torch.float, requires_grad=True).view(1, -1)
        act, value = actor(state), critic(state)

        action = Categorical(act).sample()
        next_state, reward, done, info = env.step(action.item())

        log_prob = Categorical(act).log_prob(action).unsqueeze(0)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float))
        masks.append(torch.tensor([1-done], dtype=torch.float))

        state = next_state

        if done:
            break

    print("epochs: ", _, "length: ", len(rewards));

    next_state = torch.FloatTensor(next_state)
    next_value = critic(next_state)

    # rewards
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + param.discount * R * masks[step]
        returns.insert(0, R)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    actor_optim.zero_grad()
    critic_optim.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    actor_optim.step()
    critic_optim.step()

torch.save(actor.state_dict(), 'actor.pt')
torch.save(critic.state_dict(), 'critic.pt')