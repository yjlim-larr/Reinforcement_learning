# Summary  
  Policy gradient methods requires large number of samples, and it is the main cause of bias and high variance. In this paper, it suggests policy gradient estimator which reduces variance while it maintains acceptable bias. And we call this estimation schema as GAE. This paper's contributions are 1) justification and intuition about effective variance-reduction scheme. 2) Using trust region optimization method to value function.  
 
 ## Preliminaries  
 Terms and standart policy gradient estimate method is given on previous papers. In sutton PG's for getting non biased approximated q function, it should satisfy compatible condition. But in this paper, it does not assume compatible condition, and discuss how to obtain biased(but not too biased) estimator of Advantage function.   

 It introduces a parameter γ that allows it to reduce variance by downweighting rewards corresponding to delayed effects, at the cost of introducing bias. They treat it as a variance reduction parameter in an undiscounted problem; this technique was analyzed theoretically by Marbach & Tsitsiklis (2003); Kakade (2001b); Thomas (2014).  
 
 It uses disctouned advantage, value, q functions and use it for defining discounted approximation to the policy gradient. For obtaining biased(but not too biased) estimator of Advantage function, they introduce the notion of a γ-just estimator of the advantage function, which is an estimator that does not introduce bias when we use it in place of Aπ,γ (which is not known and must be estimated) in Equation if policy gradient to estimate g^γ. **(No bias means E[Y - predicted(Y)] = 0)**  
 
 The definition is as follows  
 
 
   
   
# Result  

# Reference  
HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION: https://arxiv.org/pdf/1506.02438.pdf  
