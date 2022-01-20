# Summary  
Standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically).  

___   
1. Sample Complexity: 학습에 사용된 sample들의 다양성 정도
2. Robustness: 모수의 변화에 민감하게 반응하는 정도
3. Data efficiency: 얻어진 data sample에서 얻어내는 정보의 양의 정도
4. scalable: 모델을 키울수록 다른것들이 더 추가되어야 하는 정도

## Introduction
Although many reinforcement learning methods are considered so far, still there is room for improvement in developing a method that is scalable (to large models and parallel implementations), data efficient, and robust (i.e., successful on a variety of problems without hyperparameter tuning). Q-learning (with function approximation) fails on
many simple problems and is poorly understood, vanilla policy gradient methods have poor data effiency and robustness; and **trust region policy optimization (TRPO) is relatively complicated, and is not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks). This paper seeks to improve the current state of affairs by introducing an algorithm that attains the data efficiency and reliable performance of TRPO, while using only first-order optimization.**  
1. We propose a novel objective with clipped probability ratios, which forms a pessimistic estimate (i.e., lower bound) of the performance of the policy  
2.  To optimize policies, we alternate between sampling data from the policy and performing several epochs of optimization on the sampled data  

## Background  
1. **Policy gradient Methods:**  
2. **Trust region methods:**  
  In TRPO, objective function is approximated linear, and constraint is approximated quadratic form. So we can use conjugate gradient algorithm for calculating step update direction. And by using Largrange Multiplier Methods, constraint term becomes penalty terms, in objective function. It becomes easier to solve. **But, it is difficult to determine step size.(It does not use fixed penalty coefficient 'beta'). Beta is determined by line searching on constraint, so it is impossible to use it fix.** For achieving our goal of a first-order algorithm that emulates the monotonic improvement of TRPO, experiments show that it is not sufficient to simply choose a fixed penalty coefficient β and optimize the penalized objective Equation (5) with SGD; additional modifications are required.


## Clipped Surrogate Objective  
Let's set a notation. 
<p align="center"> <img src="./img/notation.png" alt="rewrite" width="70%" height="70%"/> </p>   

TRPO's surrogate objective function is as follows and, its goal is to maximize it.  
<p align="center"> <img src="./img/TRPO.png" alt="rewrite" width="55%" height="55%"/> </p>   

Without a constraint, maximization of L_CPI would lead to an excessively large policy update; hence, we now consider how to modify the objective, to penalize changes to the policy that move r_t(θ) away from 1. Because updated policy gives high probabiltiy on specific state that old policy's probability is too small, trpo's surrogate objective function's value diverges to infinite regardless of advantage function.  

The main objective we propose is the following:  
<p align="center"> <img src="./img/main.png" alt="rewrite" width="60%" height="60%"/> </p>   

___  
**Q. Why is this form suggested?**  
If r_t(θ) is out of range [1-ε, 1+ε], it removes incentive. Finally, (7) is L_CPI's lower bounds. If θ is equal to θ_old, L_CLIP = L_CPI. But θ becomes far away from θ_old, L_CLIP becomes far away from L_CPI. **Original TRPO's surrogate function's "constraint role(=function)" is equal to clipped objective fuction's "clip".** So we does not need to use quadratic form of constraint in L_CLIP!  
___   

Figure 1 shows how L_CLIP is drawn. **Figure 2 shows L_CLIP is less than L_CPI.(=lower bound of L_CPI is L_CLIP)** It means that it gives penalty on large policy step!) This prevents L_CPI from getting too big.  

## Adaptive KL Penalty Coefficient  
**NOTE: It is different with L_CLIP. It is another method that use KL diverge.** Another approach, which can be used as an alternative to the clipped surrogate objective, or in addition to it, is to use a penalty on KL divergence, and to adapt the penalty coefficient so that we achieve some target value of the KL divergence d_targ each policy update. In our experiments, we found that 1) the KL penalty performed worse than 2) the clipped surrogate objective, however, we’ve included it here because it’s an important baseline.  
<p align="center"> <img src="./img/KL.png" alt="rewrite" width="70%" height="70%"/> </p>   
"d" means distance between old policy and updated policy. The Large d means that it is updated large step.  
d < d_targ / 1.5( = small d = step is small = we should extend update size. So reduce penalty) d_targ is hyperparameter and initial beta(penalty coefficient is hyperparameter.)   
## Algorithm  



# Results  


# Reference  
Proximal Policy Optimization Algorithms: https://arxiv.org/pdf/1707.06347.pdf
