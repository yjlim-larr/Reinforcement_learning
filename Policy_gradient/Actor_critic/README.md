# Paper
Actor-Critic Algorithms: https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf

# Summary
These are two-time-scale algorithms in which the critic uses TD learning with a linear approximation architecture and the actor is updated in an approximate gradient direction based on information provided by the critic.  

**Actor-only methods**: A possible drawback of such methods is that the gradient estimators may have a large variance. It means policy gradient has larg error.  

**Critic-only methods**:  Such methods are indirect in the sense that they do not try to optimize directly over a policy space. Its result depends largely on policy function's quality.  

**Actor-critic methods**:  Combining the strong points of actor-only and criticonly methods, it holds the promise of delivering faster convergence (due to variance reduction), when compared to actor-only methods and critic only methods. 





# Results

# Problem

# Reference
