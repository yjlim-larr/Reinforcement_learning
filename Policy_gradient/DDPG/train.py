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

# optimizer
critic_optim = optim.Adam(critic.parameters(), lr = param.critic_lr)
actor_optim = optim.Adam(actor.parameters(), lr = param.actor_lr)

for k in range(param.episodes):
    prev_state = env.reset()
    prev_state = torch.tensor(prev_state, dtype=torch.float, requires_grad=True).view(1, -1)
    pre_noise = torch.zeros((1,1))

    for i in range(200):
        # get_action 1
        # action = (actor(prev_state) + torch.rand((1, 1)) * 0.5).clip(min = action_min, max = action_max)

        # get_action 2
        #noise = utils.ou_noise(pre_noise, dim=1)
        #pre_noise = noise
        #action = actor(prev_state)
        #action = torch.clip(action + noise, action_min, action_max)

        #get_action 3
        action = torch.rand((1,1)) * 4 - 2

        next_state, reward, done, _ = env.step(action.detach().numpy())

        next_state = torch.tensor(next_state, dtype=torch.float, requires_grad=True).view(1, -1)
        action = torch.tensor(action, dtype=torch.float, requires_grad=True).view(1, -1)
        reward = torch.tensor((reward + 10) / 10, dtype=torch.float, requires_grad=False).view(1, -1)
        done = torch.tensor(done, dtype=torch.float, requires_grad=False).view(1, -1)
        replay_mem.push(prev_state, action, next_state, reward, done)

        if len(replay_mem) > 1000 :
            batch_sample = replay_mem.random_sample(param.batch_size)
            states, actions, next_states, rewards, dones = utils.construct_batch(batch_sample)

            # update_critic
            next_action = target_actor(next_states).view(param.batch_size, -1)
            ans = rewards + param.gamma * target_critic(next_states, next_action.detach())
            pred_ans = critic(states, actions.detach())

            loss = torch.pow(ans - pred_ans, 2).mean()
            #print("critic_loss = ", loss)

            critic_optim.zero_grad()
            loss.backward()

            critic_optim.step()

            # update_actor
            actions = actor(states).view(param.batch_size, -1)
            loss = -critic(states, actions).sum()
            #print("average_reward = ", J)

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()

            l = len(critic.model)
            for i in range(l):
                if type(critic.model[i]) == nn.Linear:
                    target_critic.model[i].weight.data = (1 - param.soft_gradient_update) * target_critic.model[
                        i].weight + param.soft_gradient_update * critic.model[i].weight
                    target_critic.model[i].bias.data = (1 - param.soft_gradient_update) * target_critic.model[
                        i].bias + param.soft_gradient_update * critic.model[i].bias

            l = len(actor.model)
            for i in range(l):
                if type(actor.model[i]) == nn.Linear:
                    target_actor.model[i].weight.data = (1 - param.soft_gradient_update) * target_actor.model[
                        i].weight + param.soft_gradient_update * actor.model[i].weight
                    target_actor.model[i].bias.data = (1 - param.soft_gradient_update) * target_actor.model[
                        i].bias + param.soft_gradient_update * actor.model[i].bias

        prev_state = next_state


    # evaluation
    if len(replay_mem) > 1000 and k % 2 == 0:
        with torch.no_grad():
            prev_state = env.reset()
            prev_state = torch.tensor(prev_state, dtype=torch.float).view(1, -1)
            E = target_critic(prev_state, actor(prev_state))
            total = 0;

            for i in range(200):
                #env.render()
                prev_state = torch.tensor(prev_state, dtype=torch.float).view(1, -1)
                action = target_actor(prev_state)

                next_state, reward, done, _ = env.step(action.detach().numpy())
                prev_state = next_state
                total = param.gamma * total + reward

            print("episodes: ", k, "estimate: ", E.item(), "total: ", total)


    # save
    if k % 10 == 0:
        torch.save(target_actor.state_dict(), 'actor.pt')
        torch.save(target_critic.state_dict(), 'critic.pt')

env.close()
