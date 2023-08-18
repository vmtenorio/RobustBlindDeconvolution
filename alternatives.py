import cvxpy as cp
import numpy as np

from scipy.linalg import khatri_rao


def ye_eusipco_18(S, Y):

    N, M = Y.shape

    assert np.all(S == S.T), "Non-symmetric S"

    lamb, V = np.linalg.eigh(S)

    operation = khatri_rao(Y.T @ V, V)
    assert (M*N, N) == operation.shape

    g_hat = cp.Variable(N)

    obj = cp.sum(cp.abs(operation @ g_hat))

    constraints = [cp.sum(g_hat) == 1]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        print("Solver Error")
        return None, None

    Hinv_hat = V @ np.diag(g_hat.value) @ V.T
    X_hat = Hinv_hat @ Y

    return Hinv_hat, X_hat


def segarra_tsp_17(S, Y, L, tau):

    N, M = Y.shape

    assert np.all(S == S.T), "Non-symmetric S"

    lamb, V = np.linalg.eigh(S)

    psi = np.fliplr(np.vander(lamb, L))

    op1 = khatri_rao(psi.T, V)
    operation = np.kron(np.eye(M), op1.T)

    y_hat = (V.T @ Y).flatten(order='F')

    Z = cp.Variable((M, N, L))

    Z_v = cp.reshape(Z, (M*N, L))

    term_sparsity = 0
    for m in range(M):
        term_sparsity += cp.mixed_norm(Z[m,:,:], 2, 1)

    obj = cp.norm(Z_v, 'nuc') + tau*term_sparsity

    # Need to check how to vectorize Z
    constraints = [y_hat == operation @ cp.vec(Z, order='F')]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        print("Solver Error")
        return None, None
    
    return None