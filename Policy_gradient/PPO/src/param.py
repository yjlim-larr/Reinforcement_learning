episodes = 2000
batch_size = 200
capacity = 2000
tol = 1e-11

# Hidden
hidden = [64, 64]

# conservative policy iteration
alpha = 0.1

# GAE
gae_epsilon = 0.9
gamma = 0.99
lam = 0.98

# L_CLIP
ppo_epsilon = 0.05

# L_KLPEN
d_targ = 0.01
beta = 1

