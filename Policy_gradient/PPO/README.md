# Summary  
Standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically).

## Introduction
Although many reinforcement learnign methods are considered so far, still there is room for improvement in developing a method. **This paper seeks to improve the current state of affairs by introducing an algorithm that attains the data efficiency and reliable performance of TRPO, while using only first-order optimization.**  
1. We propose a novel objective with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy  
2.  To optimize policies, we alternate between sampling data from the policy and performing several epochs of optimization on the sampled data  

## Background  
1. Policy gradient Methods
2. Trust region methods  
  In TRPO, objective function is approximated linear, and constraint is approximated quadratic form. So we can use conjugate gradient algorithm for calculating step update direction. And by using Largrange Multiplier Methods, constraint term becomes penalty terms, in objective function. It becomes easier to solve. **But, it is difficult to determine step size.(It does not use fixed penalty coefficient 'beta'). Beta is determined by line searching on constraint, so it is impossible to use it fix.** For achieving our goal of a first-order algorithm that emulates the monotonic improvement of TRPO, experiments show that it is not sufficient to simply choose a fixed penalty coefficient β and optimize the penalized objective Equation (5) with SGD; additional modifications are required.


## Clipped Surrogate Objective  


# Results  


# Reference  
Proximal Policy Optimization Algorithms: https://arxiv.org/pdf/1707.06347.pdf
