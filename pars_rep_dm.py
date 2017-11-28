import numpy as np
from numpy.linalg import lstsq
from scipy.spatial import distance
## Parsimonious representation of DM,
## Reference:
##  https://arxiv.org/abs/1505.06118
## and main matlab code :
##  http://ronen.net.technion.ac.il/files/2016/07/DsilvaACHA.zip

def local_lin_reg(y, x, eps):
    # Local linear regression
    n = x.shape[0]
    dist = distance.cdist(x, x, 'sqeuclidean')
    eps_scale = np.median(dist) / eps
    #print("eps is", eps, "eps_scale", eps_scale)
    W = np.exp(-dist / (eps_scale*eps_scale))
    L = []
    ones_mat = np.ones((x.shape[0],1))
    counter = 0
    #print("LLR n is ", n)
    print("Local reg num of iterations is ", n)
    for i in range(n):
        aux = x - np.tile(x[i,:],(n,1))
        #print("x is ", x)
        #print("Aux is ", aux.shape)
        H = np.hstack((ones_mat, aux))
        #print("Shape of H is ", H.shape)
        #print("Matrix H is ",H)
        #print("Xx shape is ", H.shape)
        aux2 = np.tile(W[i,:],(H.shape[1],1))
        G = H.T * aux2
        #print("Size of G is ", G.shape)
        A = lstsq(G @ H , G)[0]
        #print("Size of A is ", A.shape)
        L.append(A[1,:])
        counter +=1
        #print("Sper Counter is ", counter)
    L = np.asarray(L)
    fx = L @ y

    res = np.sqrt(np.mean(np.square(y-fx)))
    return res

def compute_res(V, eps):
    #print("Shape of V is ",V.shape)
    n = V.shape[1]
    res = np.zeros((n,1))
    res[0] = 1
    counter = 0
    #print("n is ", n)
    print("Comput_res num of iterations is ",n)
    for i in range(1, n):
        #print("Iteration i is ", i)
        tt = V[:,:i]
        print("tt has shape ", tt.shape)
        rr = V[:,i]
        print("rr vector shape is ", rr.shape)
        res[i] = local_lin_reg(rr, tt, eps)
        counter+=1
        #print("Couner is", counter)
    return res