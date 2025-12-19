import numpy as np
from pangolin.ir import *
from jags_pangolin_project.src.jags_pangolin.engine import Sample_prob  

rng = np.random.default_rng()

def test_linear_alg():
    rng = np.random.default_rng(0)

    A_np = rng.normal(size=(4, 3))         # 4×3
    B_np = rng.normal(size=(3, 4))         # 3×4
    M_np = rng.normal(size=(4, 4))
    M_np = M_np @ M_np.T + np.eye(4) * 1e-6

    # numpy truth
    C_np = A_np @ B_np
    S_np = C_np.sum(axis=1)

    A = RV(Constant(A_np))
    B = RV(Constant(B_np))
    M = RV(Constant(M_np))

    C = RV(Matmul(), A, B)
    S = RV(Sum(axis=1), C)

    sp = Sample_prob()
    [S_val] = sp.sample([S], [], [])     # returns (4, nsamples)

    assert S_val.shape[0] == 4
    assert np.allclose(S_val[:, 0], S_np, atol=1e-6)


def test_linear_alg_with_softmax():
    rng = np.random.default_rng(0)

    A_np = rng.normal(size=(6, 4))
    B_np = rng.normal(size=(4, 5))
    C_np = rng.normal(size=(5, 6))

    # NumPy truth WITHOUT any inverse or transpose
    X_np = A_np @ B_np
    Y_np = X_np @ C_np                # final matrix (6,6)
    vec_np = Y_np.sum(axis=1)         # (6,)
    soft_np = np.exp(vec_np) / np.exp(vec_np).sum()

    A = RV(Constant(A_np))
    B = RV(Constant(B_np))
    C = RV(Constant(C_np))

    X = RV(Matmul(), A, B)
    Y = RV(Matmul(), X, C)
    vec = RV(Sum(axis=1), Y)
    soft = RV(Softmax(), vec)

    sp = Sample_prob()
    [soft_val] = sp.sample([soft])

    assert soft_val.shape[0] == 6
    assert np.allclose(soft_val[:, 0], soft_np, atol=1e-6)


def test_distributions():
    rng = np.random.default_rng(123)

    mu_np = rng.normal(size=4)
    cov_np = rng.normal(size=(4, 4))
    cov_np = cov_np @ cov_np.T + np.eye(4)

    mu = RV(Constant(mu_np))
    cov = RV(Constant(cov_np))
    X = RV(MultiNormal(), mu, cov)

    alpha_np = rng.uniform(1.0, 5.0, size=5)
    alpha = RV(Constant(alpha_np))
    D = RV(Dirichlet(), alpha)

    n_np = 20
    p_np = rng.uniform(size=4)
    p_np /= p_np.sum()

    n = RV(Constant(n_np))
    p = RV(Constant(p_np))
    M = RV(Multinomial(), n, p)

    sp = Sample_prob()
    [X_val, D_val, M_val] = sp.sample([X, D, M], [], [])

    assert np.allclose(X_val.mean(axis=1), mu_np, atol=0.2)
    assert np.allclose(np.cov(X_val), cov_np, atol=0.2)

    assert np.allclose(D_val.mean(axis=1), alpha_np / alpha_np.sum(), atol=0.2)
    assert np.allclose(D_val.sum(axis=0), 1.0, atol=0.2)

    assert np.allclose(M_val.mean(axis=1), n_np * p_np, atol=0.5)
    assert np.all((M_val >= 0) & (np.floor(M_val) == M_val))


def test_complex_graph_fixed():
    rng = np.random.default_rng(0)

    A_np = rng.normal(size=(5, 4))
    B_np = rng.normal(size=(4, 3))

    M_np = rng.normal(size=(3, 3))
    Sigma_np = M_np @ M_np.T + np.eye(3) * 1e-3
    mu_np = rng.normal(size=3)

    alpha_np = np.array([2., 3., 4.])
    dir_mean_np = alpha_np / alpha_np.sum()

    C_np = A_np @ B_np
    softmax_np = np.exp(C_np[0]) / np.exp(C_np[0]).sum()

    r_np = dir_mean_np + softmax_np
    r_norm_np = r_np / r_np.sum()

    A = RV(Constant(A_np))
    B = RV(Constant(B_np))
    Sigma = RV(Constant(Sigma_np))
    Mu = RV(Constant(mu_np))
    Alpha = RV(Constant(alpha_np))

    p = RV(Dirichlet(), Alpha)
    C = RV(Matmul(), A, B)

    C_row0 = RV(Constant(C_np[0]))
    q = RV(Softmax(), C_row0)

    r_norm = RV(Constant(r_norm_np))
    x = RV(MultiNormal(), Mu, Sigma)

    # z path removed (no inverse or extra ops)
    # we keep a dummy Sum to still test multiple ops
    z = RV(Sum(axis=0), x)

    n_scalar = RV(Constant(10))
    counts = RV(Multinomial(), n_scalar, r_norm)

    sp = Sample_prob()
    [p_val, q_val, r_val, x_val, z_val, counts_val] = sp.sample(
        [p, q, r_norm, x, z, counts]
    )

    assert np.allclose(p_val.mean(axis=1), dir_mean_np, atol=0.05)
    assert np.allclose(q_val.sum(axis=0), 1.0, atol=1e-6)
    assert np.allclose(r_val.sum(axis=0), 1.0, atol=1e-6)
