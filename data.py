import numpy as np

from utils.gsp_utils import *

def gen_data(N, M, g_params, p_n, eps, K=4, sparsity=2, neg_coefs=True, exp_coefs=False, coef=1, sort_h=False, norm_S=False, norm_H=False, pert_type="rewire", creat=None, dest=None, sel_ratio=1, sel_node_idx=0, x_model="ones", p_x=0.1, seed=None):
    #Generating graph and graph filter
    S = generate_graph(N, g_params, seed=seed)
    N = S.shape[0] # When using Zachary's karate club, N is ignored, thus setting it here again

    # Generate graph filter
    H, G, h = generate_graph_filter(S, K, neg_coefs, exp_coefs, coef, sort_h, norm_S, norm_H, return_G=True, return_h=True)

    # Perturbate adjacency
    Sn = pert_S(S, pert_type, eps, creat, dest, sel_ratio, sel_node_idx)

    # Generating data samples
    if x_model == "ones":
        X = np.zeros((N,M))
        for i in range(M):
            idxs = np.random.permutation(N)[:sparsity]
            X[idxs,i] = 1.# / sparsity
    elif x_model == "bernoulli_gaussian":
        X = (np.random.rand(N,M)<p_x).astype(int)
        X = X * np.random.randn(N,M)

    Y = H@X

    # Adding noise
    norm_y = (Y**2).sum() / M
    Y += np.random.randn(N,M) * np.sqrt(norm_y*p_n / N)

    return X, Y, H, G, S, Sn, h