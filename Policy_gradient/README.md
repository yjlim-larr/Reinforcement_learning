# What is Policy gradient method?
## Terms
<p align="center"> <img src="./img/terms.png" alt="MLE" width="80%" height="80%"/> </p>

## Purpose
Finding optimal policy funtion's parameter theta by using policy gradient.

Update rule is that
<p align="center"> <img src="./img/update.png" alt="MLE" width="20%" height="20%"/> </p>

Using optimal policy function, and finding optimal trajectory is the ultimate goal of policy gradient method.  
**NOTE: But it is difficult to calculate policy gradient so, we should estimate it. But "Sutton PG" suggests how to estimate it**

# Monte carlo policy gradient
It uses trajectories data for estimating policy gradient. Trajectories data is obtained from policy function by simulation. But each trajectories has different length and, if some trajectories are long, it takes a long time to learn. And use various trajectories makes estimated policy gradient have high variace. 

**NOTE** : In monte carlo simulation, variance means error. For example, let's consider situation where we got sampled trajectories data from policy function. We can estimate 
E[H(X)] by using monte carlo simulation.
<p align="center"> <img src="./img/monte.png" alt="MLE" width="30%" height="30%"/> </p>

If we define error by using mean square error,
<p align="center"> <img src="./img/MSE.png" alt="MLE" width="30%" height="30%"/> </p>

Estimated result from sampled trajectories data is equal to sample mean, we can rewrite that form like 
<p align="center"> <img src="./img/rewrite.png" alt="rewrite" width="60%" height="60%"/> </p>

and, sample mean's mean is unbiased estimate of expected value, so  
<p align="center"> <img src="./img/results.png" alt="MLE" width="80%" height="80%"/> </p>

We can conclude that monte carlo simulation's result's variance is equal to MSE.

# Actor-critic policy gradient
It is a kind of Policy gradient method. It uses two weight parameter for representing actor and critic. Critic's concept is to complement for not known action-value function.

**Actor**: At actor step, it uses critic step's result's approximated Q-function for evaluating policy, and use that result for updating policy. For updating policy, it use policy gradient for deciding which direction to update.  

**Critic**: At critic step, it uses actor step's result's policy function for approximating Q-function(action-value function) which makes present policy function optimal.    

# Sampling method
## Temporal difference
It is the method for traning model. **As the episode progresses, train the model by using every step's state and action pair, where action is chosen from policy function**.  
If model is defined V_theta(s_t, a_t), and it means when it is on state 's' and choose action 'a' at time 't', its value is calculated from model V_theta. If agent take action 'a' on state 's' and next state is s_next, model return reward R. The target value is defined as
<p align="center"> <img src="./img/TD.png" alt="rewrite" width="30%" height="30%"/> </p>

And model predicts value from state 's' and it is defined as
<p align="center"> <img src="./img/predict.png" alt="rewrite" width="20%" height="20%"/> </p>

Model trained from loss, and loss is defined as
<p align="center"> <img src="./img/Loss.png" alt="rewrite" width="30%" height="30%"/> </p>

## Monte carlo simulation
It is the same as Temporal difference in that it trains the model, but Monte carlo simulation use trajectories data, not step data.


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
