import torch
import torch.nn as nn
import torch.nn.functional as F


# policy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.s = state_dim
        self.a = action_dim

        self.model = nn.Sequential(
            nn.Linear(self.s, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, self.a)
        )

    def forward(self, input):
        return (F.softmax(self.model(input)))



# approximated Q
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.s = state_dim
        self.a = action_dim

        self.model = nn.Sequential(
            nn.Linear(self.s, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 1)
        )

    def forward(self, input):
        return self.model(input)