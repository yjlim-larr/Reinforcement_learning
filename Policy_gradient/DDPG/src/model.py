import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_space, action_high, action_low):
        super(Actor, self).__init__()

        self.state_size = state_space
        self.a_h = action_high
        self.a_l = action_low

        self.model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.a_h * self.model(input)



class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()

        self.state_size = state_space
        self.a = action_space

        self.model = nn.Sequential(
            nn.Linear(self.state_size + self.a, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        input = torch.cat([state, action], dim = 1)
        return self.model(input)