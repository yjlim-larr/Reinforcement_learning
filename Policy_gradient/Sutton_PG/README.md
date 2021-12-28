# Summary:
Make evaluating policy function and use its gradient for update policy. Assume poliy function is parameterized by theta, using policy gradient, update poliy function for finding 
optimal policy function.

* Each term means
<p align="center"> <img src="./img/terms.png" alt="MLE" width="80%" height="80%"/> </p>

* Using policy gradient, update policy function's parameter theta. Policy function is defined as:
<p align="center"> <img src="./img/policy.png" alt="MLE" width="30%" height="30%"/> </p> 

* Policy gradient and Update rule is:
<p align="center"> <img src="./img/update.png" alt="MLE" width="30%" height="30%"/> </p>

* In this paper suggest how to get policy gradient. Policy gradient caculated by
<p align="center"> <img src="./img/PG.png" alt="MLE" width="30%" height="30%"/> </p>

* So we need to define the function that evaluate poliy. And it is defined as
<p align="center"> <img src="./img/eval.png" alt="MLE" width="60%" height="60%"/> </p>


## Theorem 1.
<p align="center"> <img src="./img/theorem1.png" alt="MLE" width="80%" height="80%"/> </p>
  
**NOTE**: the key aspect of both expressions for the gradient is that their are no terms of the form **d^pi(s)'s derivative with respect theta** : the effect of policy changes on the distribution of states does not appear.   
**NOTE**: Also Q is defined by poliy, but its derivative with respect theta does not appear at the policy gradient. 

Q is not noramally known and must be estimated. it can be estimated by
<p align="center"> <img src="./img/reward.png" alt="MLE" width="50%" height="50%"/> </p>
It means we can estimate Q by sampling.


## Theorem 2.
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
this condition is satisfy We can get theorem 2's result: 
<p align="center"> <img src="./img/result.png" alt="MLE" width="30%" height="30%"/> </p>


## Theorem 3.

# 
