# Summary
In this paper, 
1) Transforming stochastic policy to deterministic policy
2) For exploration, use off policy actor critic algorithm.

Because of using "Deterministic policy", it uses off policy for exploration.  
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
___
So, **stochastic policy gradient(SPG)** from expected reward by stochastic policy function is defined as
<p align="center"> <img src="./img/SPG.png" alt="MLE" width="50%" height="50%"/> </p>
We can get this result from Sutton PG paper. This form shows that S.P.G is not realted with discounted state distribution's gradient. One issue that these algorithms must address is how to estimate the action-value function Qπ(s, a). Perhaps the simplest approach is to use a sample return(rewards) to estimate the value of Qπ(s_t, a_t), which leads to a variant of the REINFORCE algorithm.   

___
In **Stochastic actor-critic algorithm**, actor is updated by SPG. Instead of using Qπ(s_t, a_t), Critic approximate Qπ(s_t, a_t) to Qw(s_t, a_t) by using Temporal difference or Monte carlo simulation. But for no bias Sutton's PG, critic's parameters should satisfy 2 conditions.
<p align="center"> <img src="./img/CONDI.png" alt="MLE" width="100%" height="100%"/> </p>  
It is also proven by Sutton.  

If 1), 2) conditions are satisfied, Qπ - Qw = e is zero, so bias is zero. If "1)" is satisfied, compatible function approximator is linear to log policy probability's gradient. "2)" means solving regression model approximaing Qπ. We can do 2) by using Temporal difference(TD) to estimate value function.

____
**Off policy Actor-Critic** means behavior policy is not equal to target policy. if we define behavior policy 'beta(a|s)', SPG is defined as
<p align="center"> <img src="./img/OFFSPG.png" alt="MLE" width="50%" height="50%"/> </p>

* It does not depend on Q-function's gradient. It means at Actor updating Critic does not affect actor for actor's convergence to local optima. In other words, there is local optima at approximated policy function about fixed approximated Q function.   
    
* Both Actor and Critic are trained by using off policy methods.   

* For calculating off policy's policy gradient, not use Qπ, but TD errror which is defined as   
<p align="center"> <img src="./img/TDERROR.png" alt="MLE" width="30%" height="30%"/> </p>

### Gradients of Deterministic Policies
**Action-Value Gradients**: 
<p align="center"> <img src="./img/PROOF.png" alt="MLE" width="140%" height="140%"/> </p>
Don't need to calculate discounted stationary state distribution's gradient, but calculate Q-function's gradient.

**Deterministic policy gradient Theorem**:
"performance objective" is evaluation function of policy, and it is defined <img src="./img/OB.png" alt="MLE" width="12%" height="12%"/> and it is rewritten as
<p align="center"> <img src="./img/RE.png" alt="MLE" width="30%" height="30%"/> </p>
and its gradient is
<p align="center"> <img src="./img/REG.png" alt="MLE" width="40%" height="40%"/> </p>

**Limit of the stochastic Policy gradient**: Deterministic policy is the special case of stochastic policy. For example, if policy function's shape is Gaussian distribution, when its variance is zero, its probability is 1 at X = mean. So it can be written as
<p align="center"> <img src="./img/SPECIAL.png" alt="MLE" width="30%" height="30%"/> </p>















# Results

# Reference
