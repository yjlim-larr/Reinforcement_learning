# What is Inverse Reinforcement Learning(IRL): 
 Reinforcement learning uses rewards which are gotten from environment's predefined function. But what if that function is not existed, how we can do reinforcement learning? Many problems from real world, there are not reward functions. So we should devise the method for this situation.   
 Inversere Reinforcement learning can be one of the solution of that situation. It infers reward function from environments by using expert trajectories data. So it does not need reward functions, because it infers it. In this repository i summarize papers which are related with Inverse Reinforcement Learning(IRL), and add proofs, explanation, my insights.  
# Train Flow of IRL: 
 In Expert trajectories data expert actions are the global optimum. Therefore actor is trained to output that actions and reward function which is inferred from expert traj data give high reward that actions. 
 
# what is dIfference between Actor critic and IRL:   
 Actor critic uses two functions, but IRL use only actor funtion. IRL does not need Critic, because it uses expert traj. Expert traj data is Critic by itself. To be more detail, IRL assumes expert traj datas' state action pairs' actions always have the highest value on that states. Therefore we just do tune actor function output expert's trajectory datas' action on that state. In that process, we can get many reward functions which make expert trajs optimal.

# Maximum likelihood method for training actor  
 
 
# Paper
1. Algorithms for Inverse Reinforcement Learning: http://ai.stanford.edu/~ang/papers/icml00-irl.pdf  
2. Apprenticeship Learning via Inverse Reinforcement Learning:  
3. Maximum Margin Planning:  
4. Maximum Entropy Inverse Reinforcement Learning:  
5. Generative Adversarial Imitation Learning:   
6. Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow:  

