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



def conjugate_gradient_algo(actor, states, g, tol = 1e-10):
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



def GAE(critic, rewards, states, next_states, gamma, lam):
    # 1 step
    TD = torch.zeros_like(rewards)
    for i in range(len(rewards)):
        TD[i] = rewards[i] + gamma * critic(next_states[i]) - critic(states[i])

    # approximate Advantage
    AD = torch.zeros_like(rewards)
    AD[-1] = TD[-1]
    for i in reversed(range(len(rewards) - 1)):
        AD[i] = TD[i] + gamma * lam * AD[i]

    # normalize
    AD = (AD - AD.mean()) / AD.std()
    return AD




def compare_policy(actor, values, returns, states, old_policy, actions):
    # use "Approximately Optimal Approximate Reinforcement Learning's Lemma 2"
    mu, std = actor(torch.Tensor(states))
    m = torch.distributions.normal.Normal(loc=mu, scale=std)
    new_policy = m.log_prob(actions)

    advantages = returns - values # Q(s,a) - Critic(s) = Q(s,a) - V(s) = A(s, a)

    gain = torch.exp(new_policy - old_policy) * advantages
    gain = gain.mean()

    return gain



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

    state_dim, a_l, a_h, hidden = actor.get_info()
    old_actor = model.Actor(state_dim, a_l, a_h, hidden)
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
            print("Actor updated")
            break
        
        ratio *= 0.5
        
    if check == 0:
        actor.set_model_params(params) # back up
        print("Not valid actor update")



def train_critic2(critic, states, next_states, rewards, gamma, epsilon):
    # gamma just values
    values = torch.zeros_like(rewards)
    values[-1] = critic(next_states[-1])
    for i in reversed(range(len(rewards)-1)):
        values[i] = rewards[i] + gamma * values[i + 1]

    # cal loss
    loss = (critic(states) - values).pow(2).mean()

    # variance
    var = loss.detach() # paper reference

    # cal_gradient_of_average_rewards
    content = torch.autograd.grad(loss, critic.parameters())
    g = flat(content)


    # 1. conjugate algorithm
    # cal step direction
    step_dir = torch.zeros_like(g)
    r = g.clone()
    p = g.clone()

    rdotr = torch.dot(r, r)
    nsteps = step_dir.size()[0]

    for i in range(nsteps):
        # 2. cal fisher vector
        # cal kl
        mu = critic(states)
        std = torch.sqrt(var)

        mu_old = mu.detach()
        std_old = std.detach()

        kl = torch.log(std / std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
        kl = kl.sum(1, keepdim=True).mean()

        # diff kl
        content = torch.autograd.grad(kl, critic.parameters(), create_graph=True)  # Multi layer.
        d_kl = flat(content)

        fx = (d_kl * p).sum()
        content = torch.autograd.grad(fx, critic.parameters())  # one layer.
        Hx = flat(content)

        # get
        _Avp =  Hx + 0.1 * p  # damping


        # approximate step_dir
        alpha = rdotr / torch.dot(p, _Avp)

        step_dir += alpha * p
        r -= alpha * _Avp

        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr

        p = r + betta * p
        rdotr = new_rdotr

        if new_rdotr <= 1e-10:
            break


    # 3. step_size
    alpha = 0.0001
    param = critic.get_model_params()
    check = 0
    for i in range(10):
        # fisher vector cal
        p = g.detach();
        mu = critic(states)
        std = torch.sqrt(var)

        mu_old = mu.detach()
        std_old = std.detach()

        kl = torch.log(std / std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
        kl = kl.sum(1, keepdim=True).mean()

        # diff kl
        content = torch.autograd.grad(kl, critic.parameters(), create_graph=True)  # Multi layer.
        d_kl = flat(content)

        fx = (d_kl * p).sum()
        content = torch.autograd.grad(fx, critic.parameters())  # one layer.
        Hx = flat(content)

        # get
        Hx = Hx + 0.1 * p  # damping

        # --------------------------------------
        # constraint
        st = 1/2 * alpha * torch.dot(step_dir, Hx) * alpha

        if(st <= epsilon):
            new_param = param - alpha * step_dir
            critic.set_model_params(new_param)
            print("Critic updated")
            difference = (critic(states) - values)
            print("difference_average: ", difference.mean(), "  max difference: ", difference.max())
            check = 1
            break
        else:
            alpha *= 0.5

    if check == 0:
        print("Not valid critic update")
