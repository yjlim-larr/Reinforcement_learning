import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import param
import model



def init_weight(model):
  if isinstance(model, nn.Linear):
    torch.nn.init.xavier_uniform(model.weight)
    model.bias.data.fill_(0)

  if isinstance(model, nn.Conv2d):
    torch.nn.init.xavier_uniform(model.weight)
    model.bias.data.fill_(0)



def kl_divergence(new_actor, old_actor, states):
    mu, std = new_actor(torch.Tensor(states))

    mu_old, std_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()

    kl = torch.log(std / std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True).mean()



def flat(content):
    flat_con = []

    for i in range(len(content)):
        flat_con.append(content[i].float().view(-1))

    flat_con = torch.cat(flat_con)

    return flat_con



def fisher_vector_cal(x, actor, states):
    x = x.detach()  # Note!
    kl = kl_divergence(new_actor=actor, old_actor=actor, states = states)

    content = torch.autograd.grad(kl, actor.parameters(), create_graph=True)  # Multi layer.
    d_kl = flat(content)

    fx = (d_kl * x).sum()
    content = torch.autograd.grad(fx, actor.parameters()) # one layer.
    Hx = flat(content)

    return Hx + 0.1 * x  #damping



def conjugate_gradient_algo(actor, states, g, tol):
    x = torch.zeros_like(g)
    r = g.clone()
    p = g.clone()

    rdotr = torch.dot(r, r)
    nsteps = x.size()[0]

    for i in range(nsteps):
        _Avp = fisher_vector_cal(p, actor, states)
        alpha = rdotr / torch.dot(p, _Avp)

        x += alpha * p
        r -= alpha * _Avp

        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr

        p = r + betta * p
        rdotr = new_rdotr

        if new_rdotr <= tol:
            break

    return x




def get_returns(rewards, dones, end_val):
    returns = torch.zeros_like(rewards)
    Monte = end_val

    l = len(rewards)
    for i in reversed(range(l)):
        Monte = rewards[i] + param.discount * (1 - int(dones[i])) * Monte
        returns[i] = Monte

    returns = (returns - returns.mean()) / returns.std()
    return returns




def compare_policy(actor, values, returns, states, old_policy_probs, actions):
    # use "Approximately Optimal Approximate Reinforcement Learning's Lemma 2"
    mu, std = actor(torch.Tensor(states))
    m = torch.distributions.normal.Normal(loc=mu, scale=std)
    new_policy = m.log_prob(actions)

    advantages = returns - values # Q(s,a) - Critic(s) = Q(s,a) - V(s) = A(s, a)

    gain = torch.exp(new_policy) / old_policy_probs * advantages
    gain = gain.mean()

    return gain



def train_critic(critic, states, returns, critic_optim):
    # cal_loss
    loss = torch.pow(returns - critic(states), 2).mean()

    # train_model
    critic_optim.zero_grad()
    loss.backward()
    critic_optim.step()



def train_actor(actor, critic, states, actions, action_probs, returns, STEP_SIZE, TOL):
    # cal_average_rewards.
    average_reward = (action_probs * returns).mean()

    # cal_gradient_of_average_rewards
    content = torch.autograd.grad(average_reward, actor.parameters())
    g = flat(content)

    # cal_step_dir
    step_dir = conjugate_gradient_algo(actor, states, g, TOL)


    # cal_step_size
    Hx = fisher_vector_cal(step_dir, actor, states)
    a = torch.dot(step_dir, Hx)
    a = 2 * STEP_SIZE / a


    # process exception
    if a <= 0:
        return

    beta = torch.sqrt(a)  # a is less than zero, nan.
    print("full step = ", (beta * step_dir).mean())


    # judge whether it is vaild
    params = actor.get_model_params()

    state_dim, a_l, a_h = actor.get_info()
    old_actor = model.Actor(state_dim, a_l, a_h)
    old_actor.set_model_params(params)

    ratio = 1
    check = 0
    for i in range(8):
        new_params = params + beta * step_dir * ratio
        actor.set_model_params(new_params)
    
        # updated policy's gain with respect to old_policy
        values = critic(states)
        updated_actor_gain = compare_policy(actor, values, returns, states, action_probs.detach(), actions)
    
        kl = kl_divergence(new_actor=actor, old_actor=old_actor, states=states)
        kl = kl.mean()
    
        if kl < param.max_kl and updated_actor_gain > 0:
            check = 1
            print("updated")
            break
        
        ratio *= 0.5
        
    if check == 0:
        actor.set_model_params(params) # back up
        print("Not valid")
