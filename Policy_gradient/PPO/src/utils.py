import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import param
import model

import random
from collections import namedtuple


Pack = namedtuple('Pack', ('state', 'next_state', 'action', 'actions_prob', 'value', 'advantage', 'reward', 'done'))
class Memory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.mem = []
        self.size = 0
        self.idx = 0

        for i in range(capacity):
            self.mem.append(None)


    def push(self, *args):
        # push data to memory
        if self.size < self.capacity:
                self.size = self.size + 1

        pack = Pack(*args)
        self.idx = (self.idx+1) % self.capacity
        self.mem[self.idx] = pack


    def clear(self):
        # clear memory
        self.size = 0
        self.idx = 0


    def get_size(self):
        # get valid memory size
        return self.size


    def shuffle(self):
        # randomly shuffle memory list
        random.shuffle(self.mem)


    def sample(self, start_idx, N):
        # get memory sample by order
        ret = self.mem[start_idx: start_idx + N]
        ret = Pack(*zip(*ret))

        states = torch.cat(ret.state, dim = 0).view(N, -1)
        next_states = torch.cat(ret.next_state, dim = 0).view(N, -1)
        actions = torch.cat(ret.action, dim = 0).view(N, -1)
        action_probs = torch.cat(ret.actions_prob, dim = 0).view(N, -1)
        values = torch.cat(ret.value, dim = 0).view(N,-1)
        ads = torch.cat(ret.advantage, dim = 0).view(N, -1)
        rewards = torch.cat(ret.reward, dim = 0).view(N, -1)
        dones = torch.cat(ret.done, dim = 0).view(N, -1)

        return states, next_states, actions, action_probs, values, ads, rewards, dones




def init_weight(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0)

    if isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0)




# GAE
def GAE(critic, rewards, dones, states, next_states, gamma, lam):
    rewards = torch.cat(rewards, dim = 0)
    values = torch.zeros_like(rewards)

    # get values by Monte carlo simulation
    for i in reversed(range(len(rewards))):
        if dones[i] != 1:
            values[i] = rewards[i] + gamma * values[i + 1]
        else:
            values[i] = rewards[i] + critic(next_states[i]).detach()

    # 1 step
    TD = torch.zeros_like(rewards)
    for i in range(len(rewards)):
        TD[i] = rewards[i] + gamma * critic(next_states[i]).detach() - critic(states[i]).detach()

    # approximate Advantage
    AD = torch.zeros_like(rewards)
    AD[-1] = TD[-1]
    for i in reversed(range(len(rewards) - 1)):
        if dones[i] != 1:
            AD[i] = TD[i] + gamma * lam * AD[i + 1]
        else:
            AD[i] = TD[i]

    # normalize
    AD = (AD - AD.mean()) / AD.std()

    return values.detach(), AD.detach()



# PPO
def CLIP_train(actor, actor_optim, states, old_actions, action_probs, ads, ppo_epsilon, critic, critic_optim, values):
    # Not affect training.
    action_probs = action_probs.detach() # old policy's prob
    values = values.detach() # local answer of value
    ads = ads.detach()

    # cal critic loss
    criterion = nn.MSELoss()
    critic_loss = criterion(critic(states), values)
    print("critic_loss: ", critic_loss)

    # calculate ratio
    mu, std = actor(states)
    m = torch.distributions.normal.Normal(loc = mu, scale = std)
    new_probs = torch.exp(m.log_prob(old_actions)).view(-1, 1)
    ratio = new_probs / action_probs

    # cal actor loss
    I1 = ratio * ads
    print("average_reward: ", I1.mean())

    I2 = torch.clamp(ratio, 1 - ppo_epsilon, 1 + ppo_epsilon) * ads
    actor_loss = -torch.min(I1, I2).mean()
    print("actor_loss: ", actor_loss)

    # train critic
    critic_optim.zero_grad()
    critic_loss.backward(retain_graph=True)
    critic_optim.step()

    # train actor
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()