I read two papers 1) A Natural policy gradient, 2) Natural Gradient Works Efficiently in Learning. "Natural Gradient Works Efficiently in Learning" paper explains why use natural gradient and how to calculate it, and "A Natural policy gradient" paper apply natural gradient to policy gradient method.  

# Natural Gradient Works Efficiently in Learning
When a parameter space has a certain underlying structure, the ordinary gradient of a function does not represent its steepest direction, but the natural gradient does. Natural gradient is that it considers parameter space's base vector for steepest direction. For example, let's consider that manifold 
<p align="center"> <img src="./img/manifold.png" alt="MLE" width="50%" height="50%"/> </p>

In that data manifold, 'x1' and 'x2' are judged "close", when its distance is calculated by Euclidean space's metric function. But considering data structure, its distance should be calculated by using "red line". So we should consider data structure and, re-define its metric function.  

___  
### Natural gradient
Let S = {w ∈ Rn} be a parameter space on which a function L(w)is defined.  
1) If S is a Euclidean space with an orthonormal coordinate system w, the distance between 'w' and 'w + dw' is
<p align="center"> <img src="./img/dis1.png" alt="MLE" width="20%" height="20%"/> </p>  

2) However, when the coordinate system is nonorthonormal, the squared length is given by the quadratic form 
<p align="center"> <img src="./img/dis2.png" alt="MLE" width="30%" height="30%"/> </p> 
For example, 
<p align="center"> <img src="./img/pf2.png" alt="MLE" width="80%" height="80%"/> </p>  

if S is curved manifold, there is no orthonormal linear coordinate. G = g_ij(w) is called Riemmanian metric tensor. 
<p align="center"> <img src="./img/G.png" alt="MLE" width="80%" height="80%"/> </p>  

___  
### Steepest descent direction of a funciton L(w) at w
It is defined d by the vector dw that minimizes L(w + dw) where |dw| has a fixed length, that is, under the constraint |dw|^2 = ε^2 for a sufficiently small constant ε.

**Theorem 1**: the gradient of Riemannian space is given that,
<p align="center"> <img src="./img/Theorem1.png" alt="MLE" width="60%" height="60%"/> </p>  

**pf)**   
<p align="center"> <img src="./img/pf.png" alt="MLE" width="80%" height="80%"/> </p>  

___
### Natural gradient learning  
Let's assume that 'z' is sampled from Q(z). And input this 'z' to neural network paramterized 'w'. I(z,w) is loss function. We can define  
**L(w) = E[I(z,w)]** : risk function, average loss.  Neural network goal is to minimize L(w).  

The optimal parameter w* optimizes L(w). We can get optimal w* by using natural gradient and update rule.   
<p align="center"> <img src="./img/update.png" alt="MLE" width="30%" height="30%"/>  </p>  


# A Natural policy gradient
A Natural Policy gradient paper suggests "covariant gradient" by defining a metric based on the underlying structure of the policy. They apply natural gradient to policy gradient method and it makes model update toward choosing a greedy optimal action.  

### Natural Gradient  
In Policy gradient method, Policy gradient(PG) is calculated 
<p align="center"> <img src="./img/PG.png" alt="MLE" width="40%" height="40%"/>  </p>   

The steepest descent direction is defined as the vector d_theta that minimizes expected future discounted rewards under the constraint that the squared length |d_theta|^2 is held to a small constant. We can check its proof on "Natural Gradient Works Efficiently in Learning". Anyway, pi is probabilistic manifold, and its fisher information matrix is defined as  
<p align="center"> <img src="./img/fisher.png" alt="MLE" width="40%" height="40%"/>  </p>   

Fisher information matrix is positive definite. But why we use fisher information matrix? Because we can calculate kl-divergence gradient by using it.
<p align="center"> <img src="./img/kl-gradient.png" alt="MLE" width="90%" height="90%"/>  </p>   

Fisher information matrix is second derivative of kl-divergence. And its proof is that
<p align="center"> <img src="./img/fisher_proof.png" alt="MLE" width="90%" height="90%"/>  </p>   

So our natural policy gradient is calculated by, 
<p align="center"> <img src="./img/natural_gradient.png" alt="MLE" width="90%" height="90%"/>  </p>  

### The Natural gradient and Policy iteration
It also approximates Q^pi(s, a) to f^pi(s, a; w). For approximation, it uses compatible function approximation which is equal to Sutton PG.  

Theorem 1: It uses Mean square loss for evaluating approximated Q fuction's performance. W* which is optimal parameter makes approximated Q function is equal to Q function. So MSELoss is zero, and satisfies MSELoss's derivative with respect to w is zero.
<p align="center"> <img src="./img/T1.png" alt="MLE" width="90%" height="90%"/>  </p>

### Greedy Policy Improvement
Theorem 2:

Theorem 3:

### 

# Results


# Reference
* A Natural policy gradient paper: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1128&context=statistics_papers  
* Natural Gradient Works Efficiently in Learning paper: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf  
* image from https://parkgeonyeong.github.io/Manifold-Learning-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-(SNE,-/.)/
* Fisher information matrix:   
  https://math.stackexchange.com/questions/2239040/show-that-fisher-information-matrix-is-the-second-order-gradient-of-kl-divergenc  
  https://en.wikipedia.org/wiki/Fisher_information
* 

