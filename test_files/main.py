from jax import numpy as jnp
from pangolin.ir import *
from jags_pangolin.engine import Sample_prob


sample = Sample_prob().sample
a = RV(Constant([[1,2,3],[4,5,6],[7,8,9]]))

elems = [
    RV(Index(), a, RV(Constant(r)), RV(Constant(c)))
    for r in range(3) for c in range(3)
]
adds = [RV(Add(), e, e) for e in elems]
[a,b,c] = sample([adds[0], adds[1], adds[2]], debug=True)

# a = RV(Constant(2))
# b = RV(Constant(0))
# c = RV(Add(), a, b)
# sample = Sample_prob().sample
# [y ] = sample([c], debug=True)
# print(y[0])