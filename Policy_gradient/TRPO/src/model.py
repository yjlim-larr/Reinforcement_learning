import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_space, action_min, action_max):
        super(Actor, self).__init__()

        self.state_size = state_space
        self.a_l = action_min
        self.a_h = action_max

        self.model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )


    def get_info(self):
        # return state dim and action range
        return self.state_size, self.a_l, self.a_h;


    def get_model_params(self):
        params = []

        for i in range(len(self.model)):
            if isinstance(self.model[i], nn.Linear):
                params.append(self.model[i].weight.flatten())
                params.append(self.model[i].bias.flatten())

        params = torch.cat(params)

        return params


    def set_model_params(self, update_params):
        # set model layer's weight and bias
        idx = 0

        for params in self.model.parameters():
            params_length = len(params.view(-1))

            new_param = update_params[idx: idx + params_length]
            new_param = new_param.view(params.size())

            params.data.copy_(new_param)
            idx += params_length

        return


    def forward(self, input):
        mu = self.model(input).clip(min = self.a_l, max = self.a_h)

        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)

        return mu, std




class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()

        self.state_size = state_space

        self.model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, input):
        return self.model(input)