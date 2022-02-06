# Summary
## Abstract  
We can get reward function set which is the result of solving IRL problem, but it is difficult to choose one.   

### Precondition:  
 We think of the expert as trying to maximize a reward function that is expressible as **a linear combination of known features**. 

### Result:  
 We show that our algorithm terminates in a small number of iterations, and that **even though we may never recover the expert’s reward function,** the policy output by the algorithm will attain performance close to that of the expert, where **here performance is measured with respect to the expert’s unknown reward function.** 
 
### conclusion:  
 **It will show that their algorithm choose plausible reward function which expert uses by comparing other reward funtion in reward function set.**  



## 1. Introduction  
 We believe that even the reward function is frequently difficult to specify manually. To specify a reward function for the driving task, we would have to **assign a set of weights stating exactly** how we would like to trade off these different factors. Despite being able to drive competently, the authors do not believe **they can confidently specify a specific reward function for the task of “driving well.”**    
 we believe that, for many problems, the difficulty of manually specifying a reward function represents a significant barrier to the broader applicability of reinforcement learning and optimal control algorithms.  
 The task of learning from an expert is called apprenticeship learning (also learning by watching, imitation learning, or learning from demonstration). **Simply imitating expert trajectory by penalizing deviation from the desired trajectory, agent doesn't consider the environment, but action.** Author assume that reward funtion is the basis of reinforcement learning rather than policy and value function which is derived from it.    
 In this paper, they assume that the expert is trying (without necessarily succeeding) to optimize an unknown **reward function that can be expressed as a linear combination of
known “features.”** Even though they cannot guarantee that our algorithms will correctly recover the expert’s true reward function, **we show that our algorithm will nonetheless find a policy that performs as well as the expert, where performance is measured with respect to the expert’s unknown reward function.**
 
## 2. Preliminaries  

## 3. Algorithm  

## 4. Theoretical Result  


# Reference
* Apprenticeship Learning via Inverse Reinforcement Learning: http://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Abbeel+Ng:2004.pdf    
* 
