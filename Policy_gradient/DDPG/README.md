# Summary
In this paper, it uses deterministic policy gradient to implement actor-critic model free algorithm for continuous action space. Naive application of this actor-critic model is unstable. So it combines DQN and actor-critic approach. It uses DQN's idea, introduces neural network to DPG for implementing actor-critic approach about continuous action space.

### Background
___
**x_t**: observation, **a_t**: action, **r_t**: reward  
Environment is partially observed so, we need observation and action pair's history for expressing state as    
**s_t = {x_1, a_1, x_2, a_2, ..., a_t-1, x_t}**  

But, in this paper, assuming environment is fully observed. so **s_t** is equal to **x_t**.

Agent's action is expressed as policy, **π**: S -> P(A)  
initial state distribution **P(s_1)**, transition dynamics distribution is **P(s_t+1 | s_t, a_t)**, and reward function is **r(s_t, a_t)**.  
discounted future reward **R_t**, and actor's goal is to maximize expected discounted future reward '**J**'.

Using policy function **π**, visitation distribution for a policy **π** is expressed as **P_π(s)**.  
Action-value function is **Qπ(s_t, a_t)** = E[R_t | s_t, a_t]
___
**Bellman equation**: 

# Results
# Reference
