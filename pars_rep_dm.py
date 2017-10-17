import numpy as np
from numpy.linalg import lstsq
from scipy.spatial import distance
# Parsimonious representation of DM,
# Reference : https://arxiv.org/abs/1505.06118

def local_lin_reg(y, x, eps):
    # Local linear regression
    n = x.shape[0]
    dist = distance.cdist(x, x, 'sqeuclidean')
    eps_scale = np.median(dist) / eps
    print("eps is", eps, "eps_scale", eps_scale)
    W = np.exp(-dist / (eps_scale*eps_scale))
    L = []
    ones_mat = np.ones((x.shape[0],1))
    for i in range(n):
        aux = x - np.tile(x[i,:],(n,1))
        H = np.hstack((ones_mat, aux))
        aux = np.tile(W[i,:],(H.shape[1],1))
        G = H.T * aux
        A = lstsq(G @ H , G)[0]
        L.append(A[1,:])
    L = np.asarray(L)
    fx = L @ y

    res = np.sqrt(np.mean(np.square(y-fx)))
    return res

def compute_res(V, eps):
    n = V.shape[1]
    res = np.zeros((n,1))
    res[0] = 1
    for i in range(1, n):
        res[i] = local_lin_reg(V[:,i], V[:,:i], eps)
    return res