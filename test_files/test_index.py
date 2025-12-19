import numpy as np
from pangolin.ir import *
from jags_pangolin_project.src.jags_pangolin.engine import Sample_prob  


def test_simple_index_scalar_indices():
    x = RV(Constant([[1,2,3],
                     [4,5,6]]))
    i = RV(Constant(1))
    j = RV(Constant(2))
    z = RV(SimpleIndex(), x, i, j)

    sp = Sample_prob()
    [z_samps] = sp.sample([z], [], [])
    assert np.allclose(z_samps[0], 6)


def test_simple_index_vector_and_scalar_index():
    x = RV(Constant([[1,2,3],
                     [4,5,6]]))
    i = RV(Constant([0,1]))
    j = RV(Constant(2))
    z = RV(SimpleIndex(), x, i, j)

    sp = Sample_prob()
    [z_samps] = sp.sample([z], [], [])
    expected = np.array([[3], [6]])
    assert np.allclose(z_samps[:,0], expected)


def test_simple_index_two_vectors():
    A = np.arange(6).reshape(2,3)
    x = RV(Constant(A))
    i = RV(Constant([0,1,0,1,0,1]))
    j = RV(Constant([1,2,0,2,1,0]))
    z = RV(SimpleIndex(), x, i, j)

    sp = Sample_prob()
    
    [z_val] =sp.sample([z], {})
    expected = np.array([
        [A[0,1], A[0,2], A[0,0], A[0,2], A[0,1], A[0,0]],
        [A[1,1], A[1,2], A[1,0], A[1,2], A[1,1], A[1,0]],
        [A[0,1], A[0,2], A[0,0], A[0,2], A[0,1], A[0,0]],
        [A[1,1], A[1,2], A[1,0], A[1,2], A[1,1], A[1,0]],
        [A[0,1], A[0,2], A[0,0], A[0,2], A[0,1], A[0,0]],
        [A[1,1], A[1,2], A[1,0], A[1,2], A[1,1], A[1,0]],
    ])
    assert np.allclose(z_val[:,:,0], expected)

def test_simple_index_with_two_categorical_indices():
    x = RV(Constant([[3, 4, 5],
                     [6, 7, 8]]))
    prow = RV(Constant([0.5, 0.5]))
    i = RV(Categorical(), prow)
    pcol = RV(Constant([0.2, 0.3, 0.5]))
    j = RV(Categorical(), pcol)
    z = RV(SimpleIndex(), x, i, j)

    sp = Sample_prob()
    
    [z_val] = sp.sample([z], {})
    expected_values = np.array([3,4,5,6,7,8])
    assert np.all(np.isin(z_val[0], expected_values))