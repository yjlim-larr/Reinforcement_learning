# Summary
In this paper, 
1) Transforming stochastic policy to deterministic policy
2) For exploration, use off policy actor critic algorithm.

**off policy**: Use different policies for exploration and traning. On exploration step, it uses off policy, and on train step, it uses target policy.  
 
### Introduction
Standard policy function is defined as prbability distribution, but in this paper, policy function is defined as
<p align="center"> <img src="./img/policy.png" alt="MLE" width="10%" height="10%"/> </p>

By this form, it can be rewritten policy gradient form, and shows that its policy gradient form is the special case of stochastic policy function's policy gradient where policy variance is zero.

Comparing stochastic policy's policy gradient and deterministic policy's, in the case of deterministic policy we don't need to consider action space when estimating policy  gradient because its action is determined from state.

But it does not mean not using stochastic policy, but use it for off policy at exploration step. At traning step, target policy is deterministic policy.

In this paper, by using D.P.G it brings out actor-critic-algorithm form. Approximating action-value function and updating policy parameter from approximated action-value-gradient direction. And using compatible function approximating's concept, showing approximation does not cause P.G's bias.

### Background
Agent's goal is to maximize rewards gotten by taking action. The defined expected rewards from actor and critic approximated function is
<p align="center"> <img src="./img/ER.png" alt="MLE" width="70%" height="70%"/> </p>

r(s,a) is replaced to appriximated Q function.  

So, **stochastic policy gradient(SPG)** from expected reward by stochastic policy function is defined as
<p align="center"> <img src="./img/SPG.png" alt="MLE" width="50%" height="50%"/> </p>
We can get this result from Sutton PG paper. This form shows that S.P.G is not realted with discounted state distribution's gradient. One issue that these algorithms must address is how to estimate the action-value function Qπ(s, a). Perhaps the simplest approach is to use a sample return(rewards) to estimate the value of Qπ(s_t, a_t), which leads to a variant of the REINFORCE algorithm.  

In **Stochastic actor-critic algorithm**, actor is updated by SPG. Instead of using Qπ(s_t, a_t), Critic approximate Qπ(s_t, a_t) to Qw(s_t, a_t) by using Temporal difference or Monte carlo simulation. But for using Sutton PG, critic's parameters should satisfy 2 conditions.
<p align="center"> <img src="./img/CON.png" alt="MLE" width="50%" height="50%"/> </p>












# Results

# Reference
