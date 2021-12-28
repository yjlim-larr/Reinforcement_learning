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
R is defined as "expected" reward function, because next sate is not statinory but represented by transition probabilities.  
**NOTE**: the key aspect of both expressions for the gradient is that their are no terms of the form **d^pi(s)'s derivative with respect theta**: the effect of policy changes on the distribution of states does not appear. 
<p align="center"> <img src="./img/theorem1.png" alt="MLE" width="80%" height="80%"/> </p>

Q is not noramally known and must be estimated. it can be estimated by
<p align="center"> <img src="./img/reward.png" alt="MLE" width="50%" height="50%"/> </p>
It means we can estimate Q by sampling.

## Theorem 2.



## Theorem 3.

# 
