import pytest
import numpy as np
from pangolin.ir import *
from jags_pangolin_project.src.jags_pangolin.engine import Sample_prob  
from scipy.special import gammaln

rng = np.random.default_rng()


def test_simple_calculations():
    a_ran = np.round(rng.random(), 1)
    b_ran = np.round(rng.random(), 1)
    a = RV(Constant(a_ran))
    b = RV(Constant(b_ran))
    c = RV(Add(), a, b)
    d = RV(Mul(), a, b)
    e = RV(Div(), d, a)
    f = RV(Sub(), d, b)

    sp = Sample_prob()

    sp.sample([c], {})
    coda = sp.read_coda()
    assert np.allclose(coda['v' + str(c._n)], a_ran + b_ran, atol=1e-6)

    sp.sample([d], {})
    coda = sp.read_coda()
    assert np.allclose(coda['v' + str(d._n)], a_ran * b_ran, atol=1e-6)

    sp.sample([e], {})
    coda = sp.read_coda()
    assert np.allclose(coda['v' + str(d._n + 1)], a_ran * b_ran / a_ran, atol=1e-6)

    sp.sample([f], {})
    coda = sp.read_coda()
    assert np.allclose(coda['v' + str(d._n + 2)], a_ran * b_ran - b_ran, atol=1e-6)

def test_scalar_trig_functions():
    a_ran = np.round(rng.random(), 1) * np.pi
    a = RV(Constant(a_ran))
    b = RV(Sin(), a)
    c = RV(Cos(), a)
    d = RV(Arctan(), a)
    e = RV(Arcsinh(), a)

    sp = Sample_prob()
    sp.sample([b], {})
    coda = sp.read_coda()
    assert np.allclose(coda['v' + str(b._n)], np.sin(a_ran))

    sp.sample([c], {})
    coda = sp.read_coda()
    assert np.allclose(coda['v' + str(c._n)], np.cos(a_ran))

    sp.sample([d], {})
    coda = sp.read_coda()
    assert np.allclose(coda['v' + str(d._n)], np.arctan(a_ran))

    sp.sample([e], {})
    coda = sp.read_coda()
    assert np.allclose(coda['v' + str(e._n)], np.arcsinh(a_ran))

def test_scalar_distributions():
    a_ran = np.round(rng.random(), 1)
    b_ran = np.round(rng.random(), 1)+0.1
    a_int = np.round(rng.integers(1, 3), 1)
    b_int = np.round(rng.integers(1, 3), 1)+a_int
    a = RV(Constant(a_ran))
    b = RV(Constant(b_ran))
    a_int_rv = RV(Constant(a_int))
    b_int_rv = RV(Constant(b_int))

    c = RV(Normal(), a, b)
    sp = Sample_prob()
    sp.sample([c], {})
    coda = sp.read_coda()
    c_mean = np.mean(coda['v' + str(c._n)])
    assert np.allclose(c_mean, a_ran, atol=0.2)
    c_std = np.std(coda['v' + str(c._n)])
    assert np.allclose(c_std, b_ran, atol=0.2)

    d = RV(NormalPrec(), a, b)
    sp.sample([d], {})
    coda = sp.read_coda()
    d_mean = np.mean(coda['v' + str(d._n)])
    assert np.allclose(d_mean, a_ran, atol=0.2)
    d_std = np.std(coda['v' + str(d._n)])
    assert np.allclose(d_std, 1/np.sqrt(b_ran), atol=0.2)

    e = RV(Lognormal(), a, b)
    sp.sample([e], {})
    coda = sp.read_coda()
    e_mean = np.mean(coda['v' + str(e._n)])
    assert np.allclose(e_mean, np.exp(a_ran + 0.5 * b_ran * b_ran), atol=0.5)
    e_std = np.std(coda['v' + str(e._n)])       
    assert np.allclose(e_std, np.sqrt((np.exp(b_ran * b_ran) - 1) * np.exp(2 * a_ran + b_ran * b_ran)), atol=0.5)

    f = RV(BernoulliLogit(), a)
    sp.sample([f], {})
    coda = sp.read_coda()
    f_mean = np.mean(coda['v' + str(f._n)])
    assert np.allclose(f_mean, 1/(1 + np.exp(-a_ran)), atol=0.2)

    g = RV(Beta(), a_int_rv, b_int_rv)
    sp.sample([g], {})
    coda = sp.read_coda()
    g_mean = np.mean(coda['v' + str(g._n)])
    assert np.allclose(g_mean, a_int / (a_int + b_int), atol=0.2)
    g_var = np.var(coda['v' + str(g._n)])
    assert np.allclose(g_var, (a_int * b_int) / ((a_int + b_int) ** 2 * (a_int + b_int + 1)), atol=0.2)

    alpha = np.ones(5)
    probs = rng.dirichlet(alpha)
    probs = probs / np.sum(probs)
    vec = RV(Constant(probs))
    h = RV(Categorical(), vec)
    sp.sample([h], {})
    coda = sp.read_coda()
    counts = np.bincount(coda['v' + str(h._n)], minlength=5)
    counts = counts[1:]
    freq = counts / np.sum(counts)
    assert np.allclose(freq, probs, atol=0.1)

    i = RV(Uniform(), a_int_rv, b_int_rv)
    sp.sample([i], {})
    coda = sp.read_coda()
    i_mean = np.mean(coda['v' + str(i._n)])
    assert np.allclose(i_mean, (a_int + b_int) / 2, atol=0.2)
    i_var = np.var(coda['v' + str(i._n)])
    assert np.allclose(i_var, ((b_int - a_int) ** 2) / 12, atol=0.2)
    
    N = RV(Constant(np.round(rng.integers(1, 10), 1)))
    j = RV(BetaBinomial(),N, a_int_rv, b_int_rv)
    sp.sample([j], {})
    coda = sp.read_coda()
    j_mean = np.mean(coda['v' + str(j._n)])
    assert np.allclose(j_mean, N.op.value * a_int / (a_int + b_int), atol=0.2)
    j_var = np.var(coda['v' + str(j._n)])
    assert np.allclose(j_var, N.op.value * a_int * b_int * (a_int + b_int + N.op.value) / ((a_int + b_int) ** 2 * (a_int + b_int + 1)), atol=0.2)

    k = RV(Exponential(), a)
    sp.sample([k], {})
    coda = sp.read_coda()
    k_mean = np.mean(coda['v' + str(k._n)])
    assert np.allclose(k_mean, 1/a_ran, atol=1)
    k_var = np.var(coda['v' + str(k._n)])
    assert np.allclose(k_var, 1/(a_ran * a_ran), atol=1)

    l = RV(Gamma(), a_int_rv, b_int_rv)
    sp.sample([l], {})
    coda = sp.read_coda()
    l_mean = np.mean(coda['v' + str(l._n)])
    assert np.allclose(l_mean, a_int/b_int, atol = 0.2)
    l_std = np.std(coda['v' + str(l._n)])
    assert np.allclose(l_std, np.sqrt(a_int)/b_int, atol = 0.5)

    m = RV(Poisson(), a)
    sp.sample([m], {})
    coda = sp.read_coda()
    m_mean = np.mean(coda['v' + str(m._n)])
    assert np.allclose(m_mean, a_ran, atol=0.2)
    m_var = np.var(coda['v' + str(m._n)])
    assert np.allclose(m_var, a_ran, atol = 0.2)

def test_other():
    rng = np.random.default_rng()
    a_ran = np.round(rng.random(), 2)  # random number in [-2,2] for testing
    b_ran = np.round(rng.random() * 3 + 0.1, 2)  # positive number > 0

    a = RV(Constant(a_ran))
    b = RV(Constant(b_ran))
    sp = Sample_prob()

    # Pow: a^b
    pow_rv = RV(Pow(), a, b)
    sp.sample([pow_rv], {})
    coda = sp.read_coda()
    pow_val = np.mean(coda['v' + str(pow_rv._n)])
    assert np.allclose(pow_val, a_ran ** b_ran, atol=1e-5)

    # Abs: |a|
    abs_rv = RV(Abs(), a)
    sp.sample([abs_rv], {})
    coda = sp.read_coda()
    abs_val = np.mean(coda['v' + str(abs_rv._n)])
    assert np.allclose(abs_val, abs(a_ran), atol=1e-5)

    # Exp: e^a
    exp_rv = RV(Exp(), a)
    sp.sample([exp_rv], {})
    coda = sp.read_coda()
    exp_val = np.mean(coda['v' + str(exp_rv._n)])
    assert np.allclose(exp_val, np.exp(a_ran), atol=1e-5)

    # InvLogit: 1 / (1 + exp(-a))
    invlogit_rv = RV(InvLogit(), a)
    sp.sample([invlogit_rv], {})
    coda = sp.read_coda()
    invlogit_val = np.mean(coda['v' + str(invlogit_rv._n)])
    assert np.allclose(invlogit_val, 1 / (1 + np.exp(-a_ran)), atol=1e-5)

    # Log: log(a), ensure a > 0
    a_pos = a_ran+0.1
    a_pos_rv = RV(Constant(a_pos))
    log_rv = RV(Log(), a_pos_rv)
    sp.sample([log_rv], {})
    coda = sp.read_coda()
    log_val = np.mean(coda['v' + str(log_rv._n)])
    assert np.allclose(log_val, np.log(a_pos), atol=1e-5)

    # Loggamma: log(gamma(a)), a > 0
    loggamma_rv = RV(Loggamma(), a_pos_rv)
    sp.sample([loggamma_rv], {})
    coda = sp.read_coda()
    loggamma_val = np.mean(coda['v' + str(loggamma_rv._n)])
    assert np.allclose(loggamma_val, gammaln(a_pos), atol=1e-5)

    # Logit: log(a / (1 - a)), ensure 0 < a < 1
    a_unit = RV(Constant(np.clip((a_ran + 2) / 4, 0.01, 0.99)))  # map [-2,2] -> [0.01,0.99]
    logit_rv = RV(Logit(), a_unit)
    sp.sample([logit_rv], {})
    coda = sp.read_coda()
    logit_val = np.mean(coda['v' + str(logit_rv._n)])
    assert np.allclose(logit_val, np.log(a_unit.op.value / (1 - a_unit.op.value)), atol=1e-5)

    # Step: step(a), 1 if a > 0 else 0
    step_rv = RV(Step(), a)
    sp.sample([step_rv], {})
    coda = sp.read_coda()
    step_val = np.mean(coda['v' + str(step_rv._n)])
    expected_step = 1.0 if a_ran > 0 else 0.0
    assert np.allclose(step_val, expected_step, atol=1e-5)


