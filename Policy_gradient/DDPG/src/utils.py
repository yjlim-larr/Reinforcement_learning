import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
from collections import namedtuple

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def construct_batch(batch_sample):
  states = []
  next_states = []
  rewards = []
  actions = []
  dones = []
  l = len(batch_sample)

  for i in range(l):
    sample = batch_sample[i]

    states.append(sample.state)
    actions.append(sample.action)
    rewards.append(sample.reward)
    dones.append(sample.done)
    next_states.append(sample.next_state)

  states = torch.cat(states, dim = 0)
  actions = torch.cat(actions, dim = 0)
  next_states = torch.cat(next_states, dim = 0)
  rewards = torch.cat(rewards, dim = 0)
  dones = torch.cat(dones, dim = 0)

  return states, actions, next_states, rewards, dones

def ou_noise(x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=(1,1)):
    x = torch.tensor(x)
    rho = torch.tensor(rho)
    mu = torch.tensor(mu)
    dt = torch.tensor(dt)
    sigma = torch.tensor(sigma)

    return x + rho * (mu - x) * dt + sigma * torch.sqrt(dt) * torch.randn(dim)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Replay_mem(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0
        self.mem = []

    def push(self, *args):
        if len(self.mem) < self.capacity:
            self.mem.append(None)

        self.mem[self.position] = Transition(*args)
        self.position += 1
        self.position = self.position % self.capacity

    def random_sample(self, size):
        random_set = random.sample(self.mem, size)
        return random_set

    def __len__(self):
        return len(self.mem)
