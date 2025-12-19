import numpy as np
from pangolin.ir import *
from jags_pangolin_project.src.jags_pangolin.engine import Sample_prob  

def test_vmap_handle_nonrandom_exp():
    x = RV(Constant(1.0))

    op = VMap(Exp(), in_axes=(None,), axis_size=5)
    y = RV(op, x)

    sp = Sample_prob() 
    
    [y_val] = sp.sample([y], [], [])
    y_val = y_val[:,0]
    expected = np.exp(1.0) * np.ones(5)

    assert np.allclose(y_val, expected)

def test_vmap_handle_nonrandom_exp_2d():
    x_np = np.array([1.0, 2.0])
    x = RV(Constant(x_np))
    inner = VMap(Exp(), in_axes=(None,), axis_size=5)
    op = VMap(inner, in_axes=(0,))
    y = RV(op, x)
    sp = Sample_prob()
    [y_val] = sp.sample([y], {})
    y_val = np.array(y_val)[:, :, 0]  # preserve original [:,:,0]
    expected = np.exp(x_np)[:, None] * np.ones((2,5))
    assert np.allclose(y_val, expected)


def test_vmap_handle_nonrandom_add():
    x = RV(Constant(1.0))
    y = RV(Constant(np.array([2.0, 3.0, 4.0])))
    op = VMap(Add(), in_axes=(None, 0))
    z = RV(op, x, y)
    sp = Sample_prob()
    [z_val] = sp.sample([z], {})
    z_val = np.array(z_val)[:, 0]  # preserve original [:,0]
    expected = 1.0 + np.array([2.0, 3.0, 4.0])
    assert np.allclose(z_val, expected)


def test_vmap_handle_nonrandom_add_2d():
    x = RV(Constant(1.0))
    y_np = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    y = RV(Constant(y_np))
    inner = VMap(Add(), in_axes=(None, 0))
    op = VMap(inner, in_axes=(None, 0))
    z = RV(op, x, y)
    sp = Sample_prob()
    [z_val] = sp.sample([z], {})
    z_val = np.array(z_val)[:, :, 0]  # preserve original [:,:,0]
    expected = 1.0 + y_np
    assert np.allclose(z_val, expected)


def test_vmap_normal_iid():
    mu = RV(Constant(1.0))
    sigma = RV(Constant(1.0))
    op = VMap(Normal(), in_axes=(None, None), axis_size=5)
    y = RV(op, mu, sigma)
    sp = Sample_prob()
    [y_samples] = sp.sample([y], {})
    y_samples = np.array(y_samples)
    means = np.mean(y_samples, axis=1)
    stds = np.std(y_samples, axis=1, ddof=1)
    assert np.allclose(means, np.ones(5), atol=0.1)
    assert np.allclose(stds, np.ones(5), atol=0.1)


def test_vmap_normal_non_iid():
    mu_np = np.array([1.0, 2.0, 3.0])
    sigma_np = np.array([3.0, 2.0, 1.0])
    mu = RV(Constant(mu_np))
    sigma = RV(Constant(sigma_np))
    op = VMap(Normal(), in_axes=(0, 0))
    y = RV(op, mu, sigma)
    sp = Sample_prob()
    [y_samples] = sp.sample([y], {})
    y_samples = np.array(y_samples)
    means = np.mean(y_samples, axis=1)
    stds = np.std(y_samples, axis=1, ddof=1)
    assert np.allclose(means, mu_np, atol=0.2)
    assert np.allclose(stds, sigma_np, atol=0.2)


def test_vmap_normal_iid_2d():
    mu = RV(Constant(1.0))
    sigma = RV(Constant(1.0))
    inner = VMap(Normal(), in_axes=(None, None), axis_size=5)
    op = VMap(inner, in_axes=(None, None), axis_size=3)
    y = RV(op, mu, sigma)
    sp = Sample_prob()
    [y_samples] = sp.sample([y], {})
    y_samples = np.array(y_samples)[:, :, :]  # keep full shape
    means = np.mean(y_samples, axis=2)
    stds = np.std(y_samples, axis=2, ddof=1)
    expected_means = np.ones((3,5))
    expected_stds = np.ones((3,5))
    assert np.allclose(means, expected_means, atol=0.2)
    assert np.allclose(stds, expected_stds, atol=0.2)


def test_autoregressive_eval_nonrandom():
    length = 3
    x = RV(Constant(np.array([1.1, 2.2, 3.3])))
    op = Autoregressive(Add(), length, (0,), 0)
    z = RV(Constant(0.0))
    y = RV(op, z, x)
    sp = Sample_prob()
    [y_samples] = sp.sample([y], {})
    y_samples = np.array(y_samples)[:, 0]  
    expected = np.cumsum(x.op.value)
    assert np.allclose(y_samples, expected)


def test_autoregressive_const_rv_mapped():
    length = 12
    x = RV(Constant(0.5))
    noises_np = np.random.rand(length)
    noises = RV(Constant(noises_np))
    op = Autoregressive(Normal(), length, (0,), 0)
    y = RV(op, x, noises)
    sp = Sample_prob()
    [y_samples] = sp.sample([y], {})
    innovations = y_samples[1:, :] - y_samples[:-1, :]
    emp_stds = np.std(innovations, axis=1, ddof=1)
    assert np.allclose(emp_stds, noises_np[1:], atol=0.2)


def test_autoregressive_const_rv_unmapped():
    length = 12
    sigma = 0.9
    x = RV(Constant(0.5))
    noises = RV(Constant(sigma))
    op = Autoregressive(Normal(), length, (None,), 0)
    y = RV(op, x, noises)
    sp = Sample_prob()
    [y_samples] = sp.sample([y], {})
    innovations = y_samples[1:, :] - y_samples[:-1, :]
    emp_std = np.std(innovations, ddof=1)
    assert np.allclose(emp_std, sigma, atol=0.2)


def test_autoregressive_simple_const_rv():
    length = 12
    base_op = Composite(
        2, [Constant(1), Add(), Normal()],
        [[], [0, 2], [3, 1]]
    )
    op = Autoregressive(base_op, length, (None,), 0)
    x = RV(Constant(0.5))
    sigma = 0.3
    noises = RV(Constant(sigma))
    y = RV(op, x, noises)
    sp = Sample_prob()
    [y_samples] = sp.sample([y], {})
    innovations = y_samples[1:, :] - (y_samples[:-1, :] + 1)
    emp_std = np.std(innovations, ddof=1)
    assert np.allclose(emp_std, sigma, atol=0.2)


def test_composite_add():
    op = Composite(1, [Add()], [[0, 0]])
    x = RV(Constant(1.5))
    y = RV(op, x)
    sp = Sample_prob()
    [y_val] = sp.sample([y], {})
    y_val = np.array(y_val)  # deterministic scalar
    expected = 3.0
    assert np.allclose(y_val, expected)


def test_composite_add_mul():
    op = Composite(2, [Add(), Mul()], [[0, 0], [2, 1]])
    x = RV(Constant(3.3))
    y = RV(Constant(4.4))
    z = RV(op, x, y)
    sp = Sample_prob()
    [z_val] = sp.sample([z], {})
    z_val = np.array(z_val)  # deterministic scalar
    expected = (3.3 + 3.3) * 4.4
    assert np.allclose(z_val, expected)


def test_composite_bernoulli():
    op = Composite(1, [Bernoulli()], [[0]])
    x = RV(Constant(0.7))
    y = RV(op, x)
    sp = Sample_prob()
    [y_val] = sp.sample([y], {})
    y_val = np.array(y_val)
    mean_val = y_val.mean()
    assert np.allclose(mean_val, 0.7, atol=0.1)


def test_composite_exponential():
    op = Composite(1, [Exponential()], [[0]])
    x = RV(Constant(0.7))
    y = RV(op, x)
    sp = Sample_prob()
    [y_val] = sp.sample([y], {})
    y_val = np.array(y_val)
    mean_val = y_val.mean()
    expected_mean = 1 / 0.7
    assert np.allclose(mean_val, expected_mean, rtol=0.2)


def test_composite_add_normal():
    op = Composite(2, [Add(), Normal()], [[0, 1], [2, 1]])
    x = RV(Constant(0.3))
    y = RV(Constant(0.1))
    z = RV(op, x, y)
    sp = Sample_prob()
    [z_val] = sp.sample([z], {})
    z_val = np.array(z_val)
    mean_val = z_val.mean()
    expected_mean = 0.3 + 0.1
    assert np.allclose(mean_val, expected_mean, atol=0.05)
    std_val = z_val.std(ddof=1)
    expected_std = 0.1
    assert np.allclose(std_val, expected_std, rtol=0.2)