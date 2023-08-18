import cvxpy as cp
import numpy as np


VERB = False

def step_S(Sn, H, lambd, gamma, beta, verb=VERB):
    """
    Performs the graph denoising step for the Blind Deconvolution problem
    """
    S = cp.Variable(H.shape, symmetric=True)
    d_loss = cp.sum(cp.abs(S - Sn))
    commut_loss = cp.sum_squares(H@S - S@H)
    sparsity_loss = cp.sum(cp.abs(S))

    obj = lambd*d_loss + gamma*commut_loss + beta*sparsity_loss
    constraints = [S >= 0, cp.diag(S) == 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        if verb:
            print("WARNING: Could not find optimal S -- Solver Error")
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if verb:
                print("Solver error fixed")
        except cp.SolverError as e:
            if verb:
                print("A second solver error")
                print(e)
            return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return S.value
    else:
        if verb:
            print(f"WARNING: problem status: {prob.status}")
        return None
    

def step_GX(Y, S, gamma, alpha):
    N, M = Y.shape

    G = cp.Variable((N,N), symmetric=True)
    X = cp.Variable((N,M))

    ls_loss = cp.sum_squares(G@Y - X)

    commut_loss = cp.sum_squares(G@S - S@G)

    sparsity_loss = cp.sum(cp.abs(X))
    
    # sparsity_loss = cp.sum_squares(cp.abs(X))

    obj = ls_loss + gamma*commut_loss + alpha*sparsity_loss

    #constraint_X = [cp.sum(X, 0) == np.ones(M)]
    constraint_X = [cp.trace(G) == N]

    prob = cp.Problem(cp.Minimize(obj), constraint_X)
    try:
        prob.solve()
    except cp.SolverError:
        print("Giving Solver Error")
        return None, None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return G.value, X.value

    print("Not-optimal")
    return None, None

def step_GX_rew(Y, S, Wx, gamma, alpha):

    N, M = Y.shape

    G = cp.Variable((N,N), symmetric=True)
    X = cp.Variable((N,M))

    ls_loss = cp.sum_squares(G@Y - X)
    commut_loss = cp.sum_squares(G@S - S@G)
    sparsity_loss = cp.sum(cp.multiply(Wx, cp.abs(X)))
    
    # sparsity_loss = cp.sum_squares(cp.abs(X))

    obj = ls_loss + gamma*commut_loss + alpha*sparsity_loss
    #constraint_X = [cp.sum(X, 0) == np.ones(M)]
    constraint_X = [cp.trace(G) == N]

    prob = cp.Problem(cp.Minimize(obj), constraint_X)
    try:
        prob.solve()
    except cp.SolverError:
        print("Giving Solver Error")
        return None, None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return G.value, X.value

    print("Not-optimal")
    return None, None

def step_S_rew(Sn, H, W1, W2, lambd, gamma, beta, verb=VERB):
    """
    Performs the filter identification step of the robust filter identification algorithm
    with the reweighted alternative
    """
    N = Sn.shape[0]
    S = cp.Variable((N,N), symmetric=True)

    sn_loss = cp.sum(cp.multiply(W1, cp.abs(S - Sn)))
    s_loss = cp.sum(cp.multiply(W2, cp.abs(S)))
    commut_loss = cp.sum_squares(H@S - S@H)

    obj = lambd*sn_loss + beta*s_loss + gamma*commut_loss

    constraints = [
        S >= 0,
        cp.diag(S) == 0
    ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve()
    except cp.SolverError:
        if verb:
            print("WARNING: Could not find optimal S -- Solver Error")
        try:
            prob.solve(verbose=False)
            if verb:
                print("Solver error fixed")
        except:
            if verb:
                print("A second solver error")
            return None
    except cp.DCPError:
        raise RuntimeError("Could not find optimal S -- DCP Error")

    if prob.status in ["optimal", "optimal_inaccurate"]:
        S = S.value
    else:
        if verb:
            print(f"WARNING: problem status: {prob.status}")
        return None
    
    return S


def est_GXS(Y, Sn, params, max_iters=20, th=1e-3, patience=4, rew=False, S_true=None, G_true=None, X_true=None, verbose=False):

    N, M = Y.shape

    lambda_, beta, gamma, alpha, inc_gamma = params

    err = []

    S_prev = Sn
    H_prev = Sn
    X_prev = np.zeros(Y.shape)
    S = Sn

    count_es = 0
    min_err = np.inf
    if G_true is not None:
        norm_G = np.linalg.norm(G_true)
    if S_true is not None:
        norm_S = np.linalg.norm(S_true)
    if X_true is not None:
        norm_X = np.linalg.norm(X_true)

    if rew:
        W1 = np.ones((N,N))
        W2 = np.ones((N,N))
        Wx = np.ones((N,M))
        delta1 = 1e-3
        delta2 = 1e-3
        delta3 = 1e-3

    for i in range(max_iters):
        # Filter identification problem
        if rew:
            H, X = step_GX_rew(Y, S, Wx, gamma, alpha)
        else:
            H, X = step_GX(Y, S, gamma, alpha)
        H = H_prev if H is None else H
        X = X_prev if X is None else X

        # Graph identification
        if rew:
            S = step_S_rew(Sn, H, W1, W2, lambda_, gamma, beta)
        else:
            S = step_S(Sn, H, lambda_, gamma, beta)
        if S is None:
            print("S is None")
        S = S_prev if S is None else S

        if rew:
            W1 = lambda_ / (np.abs(S - Sn) + delta1)
            W2 = beta / (S + delta2)
            Wx = alpha / (np.abs(X) + delta3)

        # Check convergence   
        # Early stopping is performed with variables error
        err_G = np.linalg.norm(G_true - H) / norm_G
        err_S = np.linalg.norm(S_true - S) / norm_S
        err_X = np.abs(X_true - X).sum() / norm_X
        err.append(err_G + err_S + err_X)
        if verbose:
            print(f"Iter: {i=}, {err_G=}, {err_S=}, {err_X=}, {err[i]=}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th and err[i] > err[i-1]:
            H_min = H
            S_min = S
            X_min = X
            i_min = i
            if verbose:
                print(f'\t\tConvergence reached at iteration {i}')
            break
        if err[i] > min_err:
            count_es += 1
        else:
            min_err = err[i]
            H_min = H.copy()
            S_min = S.copy()
            X_min = X.copy()
            i_min = i
            count_es = 0
        
        if count_es == patience:
            #print(f'\t\tES Convergence reached at iteration {i_min}')
            break

        gamma = inc_gamma*gamma if inc_gamma else gamma
        H_prev = H
        X_prev = X
        S_prev = S

    return i_min, H_min, S_min, X_min




## ITERATIVE IN 3 STEPS

def step_H(Y, X, S, gamma):
    N, M = Y.shape

    H = cp.Variable((N,N), symmetric=True)

    ls_loss = cp.sum_squares(Y - H@X)

    commut_loss = cp.sum_squares(H@S - S@H)

    obj = ls_loss + gamma*commut_loss

    prob = cp.Problem(cp.Minimize(obj))
    try:
        prob.solve()
    except cp.SolverError:
        print("Giving Solver Error")
        return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return H.value

    print("Not-optimal")
    return None


def step_X(Y, H, alpha):
    N, M = Y.shape

    X = cp.Variable((N,M))

    ls_loss = cp.sum_squares(Y - H@X)

    sparsity_loss = cp.sum(cp.abs(X))
    
    # sparsity_loss = cp.sum_squares(cp.abs(X))

    obj = ls_loss + alpha*sparsity_loss

    prob = cp.Problem(cp.Minimize(obj))
    try:
        prob.solve()
    except cp.SolverError:
        print("Giving Solver Error")
        return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return X.value

    print("Not-optimal")
    return None


def step_X_rew(Y, H, Wx, alpha):

    N, M = Y.shape

    X = cp.Variable((N,M))

    ls_loss = cp.sum_squares(Y - H@X)
    sparsity_loss = cp.sum(cp.multiply(Wx, cp.abs(X)))
    
    # sparsity_loss = cp.sum_squares(cp.abs(X))

    obj = ls_loss + alpha*sparsity_loss

    prob = cp.Problem(cp.Minimize(obj))
    try:
        prob.solve()
    except cp.SolverError:
        print("Giving Solver Error")
        return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return X.value

    print("Not-optimal")
    return None


def est_HXS(Y, Sn, params, max_iters=20, th=1e-3, patience=4, rew=False, S_true=None, H_true=None, X_true=None, verbose=False):

    N, M = Y.shape

    lambda_, beta, gamma, alpha, inc_gamma = params

    err = []

    S_prev = Sn
    H_prev = Sn
    X_prev = np.zeros(Y.shape)
    S = Sn
    X = np.ones(Y.shape)

    count_es = 0
    min_err = np.inf
    if H_true is not None:
        norm_H = np.linalg.norm(H_true)
    if S_true is not None:
        norm_S = np.linalg.norm(S_true)
    if X_true is not None:
        norm_X = np.linalg.norm(X_true)

    if rew:
        W1 = np.ones((N,N))
        W2 = np.ones((N,N))
        Wx = np.ones((N,M))
        delta1 = 1e-3
        delta2 = 1e-3
        delta3 = 1e-3

    for i in range(max_iters):
        # Filter identification problem
        H = step_H(Y, X, S, gamma)
        H = H_prev if H is None else H

        if rew:
            X = step_X_rew(Y, H, Wx, alpha)
        else:
            X = step_X(Y, H, alpha)
        X = X_prev if X is None else X

        # Graph identification
        if rew:
            S = step_S_rew(Sn, H, W1, W2, lambda_, gamma, beta)
        else:
            S = step_S(Sn, H, lambda_, gamma, beta)
        if S is None:
            print("S is None")
        S = S_prev if S is None else S

        if rew:
            W1 = lambda_ / (np.abs(S - Sn) + delta1)
            W2 = beta / (S + delta2)
            Wx = alpha / (np.abs(X) + delta3)

        # Check convergence   
        # Early stopping is performed with variables error
        err_H = np.linalg.norm(H_true - H) / norm_H
        err_S = np.linalg.norm(S_true - S) / norm_S
        err_X = np.abs(X_true - X).sum() / norm_X
        err.append(err_H + err_S + err_X)
        if verbose:
            print(f"Iter: {i=}, {err_H=}, {err_S=}, {err_X=}, {err[i]=}")
        if i > 0 and np.abs(err[i] - err[i-1]) < th and err[i] > err[i-1]:
            H_min = H
            S_min = S
            X_min = X
            i_min = i
            if verbose:
                print(f'\t\tConvergence reached at iteration {i}')
            break
        if err[i] > min_err:
            count_es += 1
        else:
            min_err = err[i]
            H_min = H.copy()
            S_min = S.copy()
            X_min = X.copy()
            i_min = i
            count_es = 0
        
        if count_es == patience:
            #print(f'\t\tES Convergence reached at iteration {i_min}')
            break

        gamma = inc_gamma*gamma if inc_gamma else gamma
        H_prev = H
        X_prev = X
        S_prev = S

    return i_min, H_min, S_min, X_min
