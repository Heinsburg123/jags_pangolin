from jax import numpy as jnp
from pangolin.ir import *
from jags_pangolin.engine import Sample_prob

a = RV(Constant([[1,2],[3,4]]))
d = RV(Constant(0))
e = RV(Constant(1))
b = RV(Index(), a, d, e)
sample = Sample_prob().sample
[y ] = sample([b])

# a = RV(Constant(2))
# b = RV(Constant(0))
# c = RV(Add(), a, b)
# sample = Sample_prob().sample
# [y ] = sample([c], debug=True)
# print(y[0])