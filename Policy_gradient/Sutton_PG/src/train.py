import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

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
X = []
Y = []
for _ in range(param.epochs):
    X.append(_)
    state = env.reset()
    log_probs = []
    rewards = []
    states = []
    next_states = []
    dones = []

    for i in count():
        #env.render()
        state = torch.tensor(state, dtype=torch.float, requires_grad=True).view(1, -1)
        states.append(state)

        act = actor(state)

        action = Categorical(act).sample()
        next_state, reward, done, info = env.step(action.item())
        log_prob = act[0][action.item()].view(1, -1).log()

        next_state = torch.tensor(next_state, dtype=torch.float, requires_grad=True).view(1, -1)

        next_states.append(next_state)
        log_probs.append(log_prob)
        rewards.append(torch.tensor([reward], dtype=torch.float))
        dones.append(torch.tensor([1-done], dtype=torch.float))

        state = next_state

        if done:
            Y.append(i)
            break

    print("epochs: ", _, "length: ", len(rewards));


    # rewards temporal difference. learning does not go well than MC
    """
    returns = []
    for i in reversed(range(len(rewards))):
        R = rewards[i] + param.discount * critic(next_states[i]) * dones[i]
        returns.insert(0, R)
    """

    # rewards Monte carlo simulation
    next_state = torch.FloatTensor(next_state)
    next_value = critic(next_state)
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + param.discount * R * dones[step]
        returns.insert(0, R)


    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    states = torch.cat(states, dim = 0)
    pred = critic(states)
    ad = returns - pred

    # train critic
    critic_loss = ad.pow(2).mean()
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # train actor
    actor_loss = -(log_probs * ad.detach()).mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()


torch.save(actor.state_dict(), 'actor.pt')
torch.save(critic.state_dict(), 'critic.pt')
env.close()

plt.plot(X, Y)
plt.xlabel("episodes")
plt.ylabel("episodes' length")
plt.show()
