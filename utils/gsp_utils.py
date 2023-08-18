import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx import is_connected, to_numpy_array, connected_watts_strogatz_graph, karate_club_graph

# Graph Type Constants
SBM = 1
ER = 2
BA = 3
SW = 4
ZACHARY = 5

MAX_RETRIES = 100

def generate_graph(N, g_params, seed=None):
    """
    Return a graph generated according to the parameters inside g_params dictionary.

    The values it must contain vary according to the type of graph to be created:
    * Erdos-Renyi Graph:
      * p: connection probability between nodes
    * Stochastic-Block Model:
      * p: connection probability between nodes of the same community
      * q: connection probability between nodes of different communities
    * Barabasi Albert:
      * m: Number of edges to attach from a new node to existing nodes

    Parameters:
        N: number of nodes of the graph
        g_params: parameters for the graph type
        seed: random number generator state
    """
    if g_params['type'] == ER:
        for _ in range(MAX_RETRIES): # MAX_RETRIES to make sure the graph is connected
            G = erdos_renyi_graph(N, g_params['p'])
            if is_connected(G):
                break
        if not is_connected(G):
            raise RuntimeError("Could not create connected graph")
    elif g_params['type'] == SW:
        G = connected_watts_strogatz_graph(N, g_params['k'], g_params['p'], MAX_RETRIES)
    elif g_params['type'] == ZACHARY:
        G = karate_club_graph()
    else:
        raise NotImplementedError("Only: ER graph available")
    return to_numpy_array(G)

def generate_graph_filter(S, K, neg_coefs=True, exp_coefs=False, coef=1, sort_h=False, norm_S=False, norm_H=False, return_h=False, return_G=False, h_coefs=None):
    """
    Generate a graph filter from the graph shift operator and random coefficients
    """
    # Generate graph filter
    if h_coefs is not None:
        h = h_coefs
    elif neg_coefs:
        h = 2 * np.random.rand(K) - 1
    else:
        h = np.random.rand(K)
    if sort_h:
        h = sorted(h, key=lambda x: np.abs(x))[::-1]
    if exp_coefs:
        h = [h[i]*np.exp(-i*coef) for i in range(K)]
    h = h / np.linalg.norm(h)
    
    eigvals_S, eigvecs_S = np.linalg.eigh(S)
    if norm_S:
        eigvals_S = eigvals_S / np.max(np.abs(eigvals_S))
    psi = np.fliplr(np.vander(eigvals_S, K))

    eigvals_H = psi @ h
    H = eigvecs_S @ np.diag(eigvals_H) @ eigvecs_S.T

    eigvals_G = 1 / eigvals_H
    eigvals_G = (eigvals_G / eigvals_G.sum()) * S.shape[0] # Make eigenvalues/trace add up to N
    #eigvals_G /= eigvals_G.sum() # Make eigenvalues/trace add up to 1
    G = eigvecs_S @ np.diag(eigvals_G) @ eigvecs_S.T

    #eigvals_H = 1 / eigvals_G
    #H = eigvecs_S @ np.diag(eigvals_H) @ eigvecs_S.T

    if norm_H:
        norm_h = np.sqrt((H**2).sum())
        H = H / norm_h
    
    if return_h and return_G:
        return H, G, h
    elif return_h:
        return H, h
    elif return_G:
        return H, G
    else:
        return H

def pert_S(S, type="rewire", eps=0.1, creat=None, dest=None, sel_ratio=1, sel_node_idx=0):
    """
    Perturbate a given graph shift operator/adjacency matrix

    There are two types of perturbation
    * prob: changes a value in the adjacency matrix with a certain
    probability. May result in denser graphs
    * rewire: rewire a percentage of original edges randomly
    """
    N = S.shape[0]

    if type == "prob":
        # Perturbated adjacency
        adj_pert_idx = np.triu(np.random.rand(N,N) < eps, 1)
        adj_pert_idx = adj_pert_idx + adj_pert_idx.T
        Sn = np.logical_xor(S, adj_pert_idx).astype(float)
    elif type == "rewire":
        # Edge rewiring
        idx_edges = np.where(np.triu(S) != 0)
        Ne = idx_edges[0].size
        unpert_edges = np.arange(Ne)
        for i in range(int(Ne*eps)):
            idx_modify = np.random.choice(unpert_edges)
             # To prevent modifying the same edge twice
            unpert_edges = np.delete(unpert_edges, np.where(unpert_edges == idx_modify))
            start = idx_edges[0][idx_modify]
            new_end = np.random.choice(np.delete(np.arange(N), start))
            idx_edges[0][idx_modify] = min(start, new_end)
            idx_edges[1][idx_modify] = max(start, new_end)
        Sn = np.zeros((N,N))
        Sn[idx_edges] = 1.
        assert np.all(np.tril(Sn) == 0)
        Sn = Sn + Sn.T
    elif type == "creat-dest":

        creat = creat if creat is not None else eps
        dest = dest if dest is not None else eps

        A_x_triu = S.copy()
        A_x_triu[np.tril_indices(N)] = -1

        no_link_i = np.where(A_x_triu == 0)
        link_i = np.where(A_x_triu == 1)
        Ne = link_i[0].size

        # Create links
        if sel_ratio > 1 and sel_node_idx > 0:
            ps = np.array([sel_ratio if no_link_i[0][i] < sel_node_idx or no_link_i[1][i] < sel_node_idx else 1 for i in range(no_link_i[0].size)])
            ps = ps / ps.sum()
        else:
            ps = np.ones(no_link_i[0].size) / no_link_i[0].size
        links_c = np.random.choice(no_link_i[0].size, int(Ne * creat),
                                replace=False, p=ps)
        idx_c = (no_link_i[0][links_c], no_link_i[1][links_c])

        # Destroy links
        if sel_ratio > 1 and sel_node_idx > 0:
            ps = np.array([sel_ratio if link_i[0][i] < sel_node_idx or link_i[1][i] < sel_node_idx else 1 for i in range(link_i[0].size)])
            ps = ps / ps.sum()
        else:
            ps = np.ones(link_i[0].size) / link_i[0].size
        links_d = np.random.choice(link_i[0].size, int(Ne * dest),
                                replace=False, p=ps)
        idx_d = (link_i[0][links_d], link_i[1][links_d])

        A_x_triu[np.tril_indices(N)] = 0
        A_x_triu[idx_c] = 1.
        A_x_triu[idx_d] = 0.
        Sn = A_x_triu + A_x_triu.T
    else:
        raise NotImplementedError("Choose either prob, rewire or creat-dest perturbation types")
    return Sn