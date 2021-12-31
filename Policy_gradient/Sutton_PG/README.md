# Summary
In this paper,
1) Average reward's gradient is proportional to policy gradient. It is the core concept of Sutton P.G
2) Approximate value function, for estimating policy gradient.

Make evaluating policy function and use its gradient for update policy. Assume poliy function is parameterized by theta, using policy gradient, update poliy function for finding 
optimal policy function.

* Each term means
<img src="./img/terms.png" alt="MLE" width="80%" height="80%"/>

* Using policy gradient, update policy function's parameter theta. Policy function is defined as:
<p align="center"> <img src="./img/policy.png" alt="MLE" width="30%" height="30%"/> </p> 

* Policy gradient and Update rule is:
<p align="center"> <img src="./img/update.png" alt="MLE" width="30%" height="30%"/> </p>

* In this paper suggest how to get policy gradient. Policy gradient caculated by
<p align="center"> <img src="./img/PG.png" alt="MLE" width="30%" height="30%"/> </p>

* So we need to define the function that evaluate poliy. And it is defined as
<p align="center"> <img src="./img/eval.png" alt="MLE" width="60%" height="60%"/> </p>


### Theorem 1:
**Differential operation which is outside the expected value go inside the expected value calculation.**
<p align="center"> <img src="./img/theorem1.png" alt="MLE" width="80%" height="80%"/> </p>
  
**NOTE**: the key aspect of both expressions for the gradient is that their are no terms of the form **d^pi(s)'s derivative with respect theta** : the effect of policy changes on the distribution of states does not appear.   
**NOTE**: Also Q is defined by poliy, but its derivative with respect theta does not appear at the policy gradient. 

Q is not noramally known and must be estimated. it can be estimated by
<p align="center"> <img src="./img/reward.png" alt="MLE" width="50%" height="50%"/> </p>
It means we can estimate Q by sampling.


### Theorem 2:
In this chapter, we can replace Q function to approximate function f with some condition. To get exact Q is difficult, so we use approximate function for getting gradient.

Q function is approximated 'f' which is parameterized 'w'.  In this case, Q is replaced by f. 'w' is trained by loss defined as (Q_pi - f_w)^2, w's gradient is 
<p align="center"> <img src="./img/gradient.png" alt="MLE" width="80%" height="80%"/> </p>

So, expected loss function on specific policy is defined as 
<p align="center"> <img src="./img/Loss.png" alt="MLE" width="60%" height="60%"/> </p>

But we don't know exactly Q, so we should use "approximated Q", not Q. Therefore w's gradient is rewritten, 
<p align="center"> <img src="./img/new_gradient.png" alt="MLE" width="80%" height="80%"/> </p>

If it is convergence to optimum, w's gradient is zero. 
<p align="center"> <img src="./img/local.png" alt="MLE" width="80%" height="80%"/> </p>

And approximated Q is unbiaed estimator of Q, expected value of approximated Q is equal to expected value of Q.
<p align="center"> <img src="./img/bias.png" alt="MLE" width="100%" height="100%"/> </p>

And if 
<p align="center"> <img src="./img/condition.png" alt="MLE" width="30%" height="30%"/> </p>
this condition is satisfy, we can get theorem 2's result: 
<p align="center"> <img src="./img/result.png" alt="MLE" width="30%" height="30%"/> </p>

**NOTE**: if "f derivative with respect of w" is zero, policy is on local or global optimal. So this case is not considered

### Application to Deriving Algorithms and Advantages:
Consider that policy function
<p align="center"> <img src="./img/policy.png" alt="MLE" width="30%" height="30%"/> </p>

For defining Theorem 2, f_w should satisfy condition. So we can get,
<p align="center"> <img src="./img/Fgradient.png" alt="MLE" width="50%" height="50%"/> </p>

If we use nonlinear f_w function, we can derive variety. Anyway above f's form is similar to "Q - V" formula
<p align="center"> <img src="./img/advantage.png" alt="MLE" width="80%" height="80%"/> </p>

Therefore, recall advantage function's property, we can get f_w's mean is zero.
<p align="center"> <img src="./img/inference.png" alt="MLE" width="80%" height="80%"/> </p>

# Using Monte Carlo simulation for estimating gradient
We can estimate gradient by using Monte Carlo simulation. 
<p align="center"> <img src="./img/monte2.png" alt="MLE" width="80%" height="80%"/> </p>

# Results
![ezgif com-gif-maker](https://user-images.githubusercontent.com/62493307/147737637-e4d9553c-2fa7-42e8-be2c-75382a9cd391.gif)

I use actor critic policy gradient and two critic train method 1) temporal difference 2) monte carlo simulation. Monte carlo simulation method shows better performance than Temporal difference, because Temporal difference has larger bias than Monte carlo i think. Cartpole is the game i use for checking my reinforcement learning model how to work.  
**NOTE** : Actor ouput is each action's probability. We should use sample() function for samping action from policy function. Not use max action probability's action.
<p align="center"> <img src="./img/graph.png" alt="MLE" width="100%" height="100%"/> </p>

It shows episodes length is convergent to "200".


