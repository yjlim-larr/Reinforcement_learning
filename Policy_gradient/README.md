# What is Policy gradient method?
## Terms
<p align="center"> <img src="./img/terms.png" alt="MLE" width="80%" height="80%"/> </p>

## Purpose
Finding optimal policy funtion's parameter theta by using policy gradient.

Update rule is that
<p align="center"> <img src="./img/update.png" alt="MLE" width="20%" height="20%"/> </p>

Using optimal policy function, and finding optimal trajectory is the ultimate goal of policy gradient method.  
**NOTE: But it has a big problem that if there is no cost funtion(or reward function), its method is hard to use.**

# Monte carlo policy gradient
It uses trajectories data for estimating policy gradient. Trajectories data is obtained from policy function by simulation. But each trajectories has different length and, if some trajectories are long, it takes a long time to learn. And use various trajectories makes estimated policy gradient have high variace. 

**NOTE** : In monte carlo simulation, variance means error. For example, let's consider situation where we got sampled trajectories data from policy function. We can estimate 
E[H(X)] by using monte carlo simulation.
<p align="center"> <img src="./img/monte.png" alt="MLE" width="30%" height="30%"/> </p>

If we define error by using mean square error,
<p align="center"> <img src="./img/MSE.png" alt="MLE" width="30%" height="30%"/> </p>

Estimated result from sampled trajectories data is equal to sample mean, we can rewrite that form like 
<p align="center"> <img src="./img/rewrite.png" alt="rewrite" width="30%" height="30%"/> </p>

and, sample mean's mean is unbiased estimate of expected value, so  
<p align="center"> <img src="./img/results.png" alt="MLE" width="80%" height="80%"/> </p>

We can get variance is equal to MSE.

# Actor-critic policy gradient
It is a kind of Policy gradient method.   
Actor:  
Critic:  

# Sampling method
## Temporal difference


## Monte carlo method


# Paper
1. Sutton PG: https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf  
  It suggests how to calculate policy gradient, so it is the basic of policy gradient method. 
  
2. Actor critic: https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
  
4. DPG: 
5. DDPG: 
6. NPG
7. TRPO
8. GAE
9. PPO

# Reference  
https://dnddnjs.gitbooks.io/rl/content/actor-critic_policy_gradient.html  
https://dnddnjs.gitbooks.io/rl/content/mc_prediction.html  
https://dnddnjs.gitbooks.io/rl/content/td_prediction.html  
https://m.blog.naver.com/yunjh7024/220863118407  
