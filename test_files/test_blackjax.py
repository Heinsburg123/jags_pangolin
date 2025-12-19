import jax.numpy as jnp
from pangolin.ir import *
from jags_pangolin_project.src.jags_pangolin.engine import Sample_prob
import numpy as np

def test_simple():
    sp = Sample_prob()

    loc = RV(Constant(0))
    scale = RV(Constant(1))
    x = RV(Normal(), loc, scale)

    [x_samps] = sp.sample([x], [], [])

    assert x_samps.ndim == 1          # Still should be a 1D sample vector
    assert jnp.abs(jnp.mean(x_samps) - 0) < 0.05
    assert jnp.abs(jnp.std(x_samps) - 1) < 0.05

def test_conditioning():
    sp = Sample_prob()

    loc = RV(Constant(0))
    scale = RV(Constant(1))
    x = RV(Normal(), loc, scale)
    y = RV(Normal(), x, scale)

    [x_samps] = sp.sample([x], [y], [1.0])

    assert x_samps.ndim == 1
    assert jnp.abs(jnp.mean(x_samps) - 0.5) < 0.05
    assert jnp.abs(jnp.var(x_samps) - 0.5) < 0.05


def test_nonrandom():
    sp = Sample_prob()

    loc = RV(Constant(0))
    scale = RV(Constant(1))
    x = RV(Normal(), loc, scale)
    y = RV(Add(), x, x)

    [x_samps, y_samps] = sp.sample([x, y], [], [])

    assert x_samps.shape == y_samps.shape
    assert jnp.allclose(y_samps, x_samps * 2)


def test_nonrandom_conditioning():
    sp = Sample_prob()

    loc = RV(Constant(0))
    scale = RV(Constant(1))
    x = RV(Normal(), loc, scale)
    z = RV(Normal(), x, scale)
    y1 = RV(Add(), x, x)
    y2 = RV(Mul(), x, x)

    [x_samps, y1_samps, y2_samps] = sp.sample(
        [x, y1, y2], [z], [1.0]
    )

    assert x_samps.shape == y1_samps.shape == y2_samps.shape
    assert jnp.abs(jnp.mean(x_samps) - 0.5) < 0.05
    assert jnp.abs(jnp.var(x_samps) - 0.5) < 0.05
    assert jnp.allclose(y1_samps, x_samps * 2)
    assert jnp.allclose(y2_samps, x_samps**2, atol=0.01)



def test_nonrandom_from_given():
    sp = Sample_prob()

    loc = RV(Constant(0))
    scale = RV(Constant(1))
    x = RV(Normal(), loc, scale)
    z = RV(Normal(), x, scale)
    y = RV(Mul(), z, scale)

    [x_samps, y_samps] = sp.sample([x, y], [z], [1.0])

    assert x_samps.shape == y_samps.shape
    assert jnp.abs(jnp.mean(x_samps) - 0.5) < 0.05
    assert jnp.abs(jnp.var(x_samps) - 0.5) < 0.05
    assert jnp.allclose(y_samps, 1.0)


def test_given_in_output():
    sp = Sample_prob()

    loc = RV(Constant(0))
    scale = RV(Constant(1))
    x = RV(Normal(), loc, scale)
    z = RV(Normal(), x, scale)
    y = RV(Mul(), z, scale)

    [y_samps, z_samps] = sp.sample([y, z], [z], [2.3])

    assert jnp.allclose(z_samps, 2.3)
    assert jnp.allclose(y_samps, 2.3)


def test_sum_axis0_deterministic():
    sp = Sample_prob()

    a_val = [
        [[1,2,3],[2,3,4]],
        [[1,2,3],[2,3,4]]
    ]
    a = RV(Constant(a_val))
    b = RV(Sum(axis=0), a)

    [b_samps] = sp.sample([b], [], [])

    expected = jnp.sum(jnp.array(a_val), axis=0)
    assert jnp.allclose(b_samps[:,:,0], expected)

def test_bernoulli():
    sp = Sample_prob()

    op = Composite(1, [Bernoulli()], [[0]])

    x = RV(Constant(0.7))
    y = RV(op, x)

    [y_samps] = sp.sample([y], [], [])
    # Properties
    assert y_samps.ndim == 1
    assert jnp.logical_or(y_samps == 0, y_samps == 1).all()
    assert abs(np.mean(y_samps) - 0.7) < 0.05


def test_exponential():
    sp = Sample_prob()

    # Composite wrapper for Exponential
    # Exponential(rate = x)
    op = Composite(1, [Exponential()], [[0]])

    x = RV(Constant(0.7))
    y = RV(op, x)

    [y_samps] = sp.sample([y], [], [])

    assert y_samps.ndim == 1
    assert (y_samps >= 0).all()
    assert abs(np.mean(y_samps) - (1.0 / 0.7))


def test_add_normal():
    sp = Sample_prob()

    op = Composite(
        2,
        [Add(), Normal()],
        [[0, 1], [2, 1]]
    )

    x = RV(Constant(0.3))
    y = RV(Constant(0.1))
    z = RV(op, x, y)

    [z_samps] = sp.sample([z], [], [])

    assert z_samps.ndim == 1
    assert abs(np.mean(z_samps) - 0.4) < 0.05

def test_handle_nonrandom_exp():
    sp = Sample_prob()
    a = RV(Constant(1.0))
    op = VMap(Exp(), [None], 5)  # VMap over 5
    y = RV(op, a)

    [y_samps] = sp.sample([y], [], [])

    expected = jnp.exp(1.0) * jnp.ones(5)
    assert jnp.allclose(y_samps[:, 0], expected, atol=0.001)  # account for sample dimension


def test_handle_nonrandom_exp_2d():
    sp = Sample_prob()
    a_val = jnp.array([1.0, 2.0])
    a = RV(Constant(a_val))
    inner = VMap(Exp(), [None], 5)
    op = VMap(inner, [0])  
    y = RV(op, a)

    [y_samps] = sp.sample([y], [], [])

    expected = jnp.exp(a_val)[:, None] * jnp.ones((2, 5))
    assert jnp.allclose(y_samps[:, :, 0], expected, atol=0.001)


def test_handle_nonrandom_add():
    sp = Sample_prob()
    x = RV(Constant(1.0))
    y_val = jnp.array([2.0, 3.0, 4.0])
    y = RV(Constant(y_val))
    op = VMap(Add(), [None, 0])  # VMap over second argument
    z = RV(op, x, y)

    [z_samps] = sp.sample([z], [], [])

    expected = x.op.value + y_val
    assert jnp.allclose(z_samps[:, 0], expected)


def test_handle_nonrandom_add_2d():
    sp = Sample_prob()
    x = RV(Constant(1.0))
    y_val = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    y = RV(Constant(y_val))
    inner = VMap(Add(), [None, 0])
    op = VMap(inner, [None, 0])  # nested VMap
    z = RV(op, x, y)

    [z_samps] = sp.sample([z], [], [])

    expected = x.op.value + y_val
    assert jnp.allclose(z_samps[:, :, 0], expected)

def test_repeated_exp_sample_prob():
    sp = Sample_prob()
    length = 5
    x = RV(Constant(0.1))
    op = Autoregressive(Exp(), length, in_axes=[])
    y = RV(op, x)

    # Compute expected deterministically
    last = 0.1
    expected = []
    for _ in range(length):
        last = np.exp(last)
        expected.append(last)
    expected = np.array(expected)

    [y_samps] = sp.sample([y], [], [])
    assert np.allclose(y_samps[:,0], expected)


def test_repeated_exp_with_dummy_sample_prob():
    sp = Sample_prob()
    length = 5
    x = RV(Constant(0.1))
    y = RV(Autoregressive(Exp(), length, in_axes=[]), x)
    dummy = RV(Normal(), x, x)

    last = 0.1
    expected = []
    for _ in range(length):
        last = np.exp(last)
        expected.append(last)
    expected = np.array(expected)

    [y_samps] = sp.sample([y], [dummy], [0.1])
    assert np.allclose(y_samps[:,0], expected)

def test_autoregressive_simple():
    sp = Sample_prob()
    x = RV(Constant(0.5))
    length = 12

    base = Composite(
        num_inputs=2,
        ops=[Constant(1.0), Add(), Normal()],
        par_nums=[[], [0, 2], [3, 1]] 
    )

    noise = RV(Constant(1e-4))
    op = Autoregressive(base, length, in_axes=[None], where_self=0)
    y = RV(op, x, noise)

    expected = 0.5 + np.arange(1, length + 1)

    [ys] = sp.sample([y], [], [])
    last_y = ys[:, 0]
    assert np.allclose(last_y, expected, atol=0.1)

def test_autoregressive_const_rv_mapped():
    sp = Sample_prob()
    x = RV(Constant(0.5))
    length = 12
    noises = RV(Constant(np.random.rand(length)))

    op = Autoregressive(Normal(), length, in_axes=(0,), where_self=0)
    y = RV(op, x, noises)

    [ys] = sp.sample([y], [], [])
    assert ys.shape[0] == length

def test_autoregressive_const_rv_unmapped():
    sp = Sample_prob()
    x = RV(Constant(0.5))
    length = 12
    noise = RV(Constant(1e-4))

    op = Autoregressive(Normal(), length, in_axes=(None,), where_self=0)
    y = RV(op, x, noise)

    [ys] = sp.sample([y], [], [])
    assert ys.shape[0] == length

def test_autoregressive_simple_const_rv():
    sp = Sample_prob()
    x = RV(Constant(0.5))
    length = 12
    noise = RV(Constant(1e-4))

    base = Composite(
        2,
        [Constant(1), Add(), Normal()],
        [[], [0, 2], [3, 1]]
    )
    op = Autoregressive(base, length, in_axes=[None], where_self=0)
    y = RV(op, x, noise)

    expected = 0.5 + np.arange(1, length + 1)

    [ys] = sp.sample([y], [], [])
    last_y = ys[:, 0]

    assert np.allclose(last_y, expected, atol=0.1)

def test_autoregressive_nonrandom():
    sp = Sample_prob()
    p1 = RV(Constant(0))
    p2 = RV(Constant(1e-5))
    x = RV(Normal(), p1, p2)
    length = 12

    op = Autoregressive(Add(), length, in_axes=[None], where_self=0)
    increment = RV(Constant(1.0))
    y = RV(op, x, increment)

    expected = np.arange(1, length + 1)

    [ys] = sp.sample([y], [], [])
    last_y = ys[:, 0]

    assert np.allclose(last_y, expected, atol=0.1)

def test_autoregressive_varying_increments():
    sp = Sample_prob()
    x = RV(Constant(0.0))
    length = 12

    increments = np.random.randn(length)
    inc_rv = RV(Constant(increments))
    noise_rv = RV(Constant(1e-4))
    base = Composite(
        num_inputs=3,                          
        ops=[Add(), Normal()],                
        par_nums=[
            [0, 1],    # Add(last, inc)
            [3, 2],    # Normal(mean, noise)
        ],
    )

    op = Autoregressive(
        base_op=base,
        length=length,
        in_axes=[0, None],
        where_self=0,
    )

    y = RV(op, x, inc_rv, noise_rv)

    expected = np.cumsum(increments)

    [ys] = sp.sample([y], [], [])
    last_y = ys[:, 0]

    assert np.allclose(last_y, expected, atol=0.1)

def test_autoregressive_matmul():
    sp = Sample_prob()
    ndim = 5
    x0 = np.random.randn(ndim)
    length = 5
    noise = RV(Constant(1e-5))
    x = RV(VMap(Normal(), in_axes=[0,None], axis_size=ndim), RV(Constant(x0)), noise)
    A = np.random.randn(ndim, ndim)
    y = RV(Autoregressive(base_op = Matmul(), length=length, in_axes=[None], where_self=1), RV(Constant(A)), x)

    expected = []
    last = x0
    for i in range(length):
        last = A @ last
        expected.append(last)

    [ys] = sp.sample([y], [], [])
    final_vals = ys[:, :,0]
    assert np.allclose(final_vals, expected, atol=0.1)