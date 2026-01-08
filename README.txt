A library that uses pangolin as an interface and utilize JAGS for backend. 
Jags_pangolin support discrete and continuous distributions, multivariate distributions, linear algebra operations(without inverse), 
vmap, autoregressive, composite and indexing. 

## Installation
```bash
pip install git+https://github.com/Heinsburg123/jags_pangolin.git
```

IMPORTANT:
You need to have already installed JAGS on your system

Example:

from jags_pangolin import run_model
from pangolin.ir import * 

a = RV(Constant([[1,2,3],[2,3,4], [3,3,3]])) 
b = RV(Constant([1,2,3])) 
c = RV(Matmul(), a, b) 
[samp] = run_model([c], [], [], ninter=10)
print(samp)

The first variable of run model are the random variables you want to monitor, the second is observed variables, the third is their observed values, and the fourth is how many iterations you want JAGS to run. 
