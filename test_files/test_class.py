from pangolin.testing.inference import InferenceTests
import pangolin.blackjax
from jax import numpy as jnp
from pangolin import ir
from jags_pangolin.engine import Sample_prob


class TestJags(InferenceTests):
    _sample_flat = Sample_prob().sample
    _cast = jnp.array
    _ops_without_eval_support = {
        ir.Inv,
        ir.Diag, 
        ir.Transpose, 
        ir.Cholesky,
        ir.Identity
    }
    _ops_without_sampling_support = {
        ir.Wishart,
    }