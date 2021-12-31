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

# Results

# Reference
