import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_space, action_low, action_high, hidden):
        super(Actor, self).__init__()

        self.state_size = state_space
        self.a_h = action_high
        self.a_l = action_low
        self.hidden = hidden

        self.model = nn.Sequential(
            nn.Linear(self.state_size, hidden[0]),
            nn.Tanh(),

            nn.Linear(hidden[0], hidden[1]),
            nn.Tanh(),

            nn.Linear(hidden[1], hidden[2]),
            nn.Tanh(),

            nn.Linear(hidden[2], 1)
        )

    def get_layer(self):
        # get layer count
        l = 0
        for i in range(len(self.model)):
            if isinstance(self.model[i], nn.Linear):
                l += 1

        return l


    def get_info(self):
        # return state dim and action range
        return self.state_size, self.a_l, self.a_h, self.hidden


    def get_model_params(self):
        # get model layer's weight and bias
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

        # std = 1
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)

        return mu, std




class Critic(nn.Module):
    def __init__(self, state_space, hidden):
        super(Critic, self).__init__()

        self.state_size = state_space

        self.model = nn.Sequential(
            nn.Linear(self.state_size, hidden[0]),
            nn.Tanh(),

            nn.Linear(hidden[0], hidden[1]),
            nn.Tanh(),

            nn.Linear(hidden[1], hidden[2]),
            nn.Tanh(),

            nn.Linear(hidden[2], 1)
        )


    def get_model_params(self):
        # get model layer's weight and bias
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
        return self.model(input)