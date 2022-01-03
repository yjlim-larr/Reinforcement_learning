# Summary
When a parameter space has a certain underlying structure, the ordinary gradient of a function does not represent its steepest direction, but the natural gradient does. Natural gradient is that it considers parameter space's base vector for steepest direction. For example, let's consider that manifold 
<p align="center"> <img src="./img/manifold.png" alt="MLE" width="50%" height="50%"/> </p>

In that data manifold, 'x1' and 'x2' is close, when its distance is calculated by Euclidean space's metric function. But considering data structure, its distance can be calculated from "red line". So we should consider data structure and, re-define its metric function.

### Natural gradient
Let S = {w ∈ Rn} be a parameter space on which a function L(w)is defined.  
1) If S is a Euclidean space with an orthonormal coordinate system w, the distance between 'w' and 'w + dw' is
<p align="center"> <img src="./img/dis1.png" alt="MLE" width="50%" height="50%"/> </p>  

2) However, when the coordinate system is nonorthonormal, the squared length is given by the quadratic form 
<p align="center"> <img src="./img/dis2.png" alt="MLE" width="50%" height="50%"/> </p> 
For example, 
<p align="center"> <img src="./img/pf.png" alt="MLE" width="50%" height="50%"/> </p>  

if S is curved manifold, there is no orthonormal linear coordinate. G = g_ij(w) is called Riemmanian metric tensor. 
<p align="center"> <img src="./img/G.png" alt="MLE" width="50%" height="50%"/> </p> 

### Steepest descent direction of a funciton L(w)
It is defined d by the vector dw that minimizes L(w + dw) where |dw| has a fixed length, that is, under the constraint |dw|^2 = ε^2 for a sufficiently small constant ε

# Results


# Reference
* A Natural policy gradient paper: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1128&context=statistics_papers  
* Natural Gradient Works Efficiently in Learning paper: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.452.7280&rep=rep1&type=pdf  
* image from https://parkgeonyeong.github.io/Manifold-Learning-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-(SNE,-/.)/
* 
