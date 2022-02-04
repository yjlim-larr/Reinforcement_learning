# Summary
## Abstract  
 This paper addresses the problem of inverse reinforcement learning (IRL) in Markov decision processes, that is, the problem of extracting a reward function given observed, optimal behavior.   
 It derives three algorithms for IRL. The first two deal with the case where the entire policy is known; it handles tabulated reward functions on a finite state space and linear functional approximation of the reward function over a potentially infinite state space. The third algorithm deal with the more realistic case in which the policy is known only through a finite set of observed trajectories. (**In reality, basically states are infinite. So it deals with infinite states set**)   
 In all cases, a key issue is degeneracy - the existence of a large set of reward functions for which the observed policy is optimal. (**It means that there are many reward functions that make expert trajectories optimal. So it is important which we choose.**) It suggests how to choose plausible(such as optimal) reward function. 
 
 **Its contributions are that**  
 1) It addresses IRL problem to computational task by using finite Markov decision processes(MDPs), and it is more familiar to the machine learning community.  
 2) they give a simple characterization of the set of all reward functions for which a given policy is optimal.  
 3) The reward function set contains degenerate solutions, so they propose a simple heuristic for removing this degeneracy, resulting in a linear programming solutiuon to the IRL problem.

## Introduction  
From charanterized IRL problem, they identify some sources of motivation.  
1) Reward function is unknown, and it can be ascertained through empirical investigation, such as bee's 
foraging behavior. We don't know how bee weights nectaringestion against various things, such as flight distance, time and risk from wind and so on.  
2) Agent designer may have only a very rough idea of optimal behaviors, so it is not useful to use straightforward reinforcement learning. We cam use epert traj data which
can be information of reward function, and use it for training, so it can be called **imitation learning** and **apprenticeship learning**. From those data, actor recover expert's reward function 
adn to use it to generate desirable behavior. So actor directly related with reward fuction, so it is the most robust definition of the task.   
(**It sees expert function as reward fuction. Therefore expert function is based on its reward function(which they thinks it is optimal)**)  

It uses finite Markov decision processes(MDPs) for addressing IRL problem to computational task and being more familiar to the machine learning community. (There were no past research, so it is the first try)   


## Notation and Problem Formulation  
<p align="center"> <img src="./img/Terms.png" alt="MLE" width="100%" height="100%"/> </p>

For discrete, finite spaces, all these functions can be represented as vectors indexed by state. 

<p align="center"> <img src="./img/Terms3.png" alt="MLE" width="100%" height="100%"/> </p>  

## Basic Properties of MDPs  
### Theorem 1(Bellman Equations)
<p align="center"> <img src="./img/1.png" alt="MLE" width="100%" height="100%"/> </p>  

### Theorem 2(Bellman Optimality)
<p align="center"> <img src="./img/2.png" alt="MLE" width="100%" height="100%"/> </p>  

## Inverse Reinforcement Learning  
The inverse reinforcement learning problem is to find a reward function that can explain observed behavior. optimal policy π is given and, they wish to find the set of possible reward functions R that makes given π is optimal. **Be careful that reward functions that satisfy given π is optimal are not unique!** For simplicity, they assume policy is deterministic.  

## IRL in Finite State Spaces  
1) they give a simple characterization of the set of all reward functions for which a given policy is optimal
2) The reward function set contains degenerate solutions, so they propose a simple heuristic for removing this degeneracy, resulting in a linear programming solutiuon to the IRL problem.

### 3.1 Characterization of the soultion set  
Why need characterization of the solution set? Because it is the reason why expert chooses those action. If we know the states' characterization, actor can choose the plausible action when it encounters strange state. It is only needed when state are infinite. Because if state space is finite, actor shows good performance when it remember all states.  

Their main result characterizing the set of solutions is the follwing:  
<p align="center"> <img src="./img/Theorem3.png" alt="MLE" width="100%" height="100%"/> </p>  

* Key points of that theorem are 
1) (I - γPa1) is always invertible (I don't understand why it has no zero eigenvalues.)
2) Remark: (Pa1−Pa)(I−γPa1)−1R≻0 is necessary and sufficient for π ≡ a1 to be unique optimal policy. 


# Reference
* Algorithms for Inverse Reinforcement Learning: http://ai.stanford.edu/~ang/papers/icml00-irl.pdf 

* 
