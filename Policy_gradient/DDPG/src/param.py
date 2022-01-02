episodes = 50
batch_size = 32
gamma = 0.95
capacity = 20000

random_noise_discount = 0.9 # use noise for exploration by Ornstein-Uhlenbeck process

weight_decay = 1e-2
actor_lr = 1e-4
critic_lr = 1e-3
soft_gradient_update = 1e-2

explore = 0.99