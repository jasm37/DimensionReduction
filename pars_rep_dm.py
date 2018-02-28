import numpy as np
import logging
import scipy.spatial.distance as dist
from scipy.linalg import lstsq

## Parsimonious representation of DM,
## Reference:
##  https://arxiv.org/abs/1505.06118
## and main matlab code :
##  http://ronen.net.technion.ac.il/files/2016/07/DsilvaACHA.zip

def local_linear_regression(Y, X, eps_med_scale):
    n = X.shape[0]
    # Compute local kernel
    K = dist.squareform(dist.pdist(X))
    eps = np.median(K)/eps_med_scale
    W = np.exp(-np.square(K)/eps**2)
    L = np.zeros((n, n))
    aux2 = np.ones((X.shape[0], 1))

    # Compute local fit for each data point
    for i in range(n):
        aux = X - np.tile(X[i, :],(n, 1))
        Xx = np.hstack((aux2, aux))
        Xx2 = Xx.T * np.tile(W[i,:], (Xx.shape[1], 1))
        # Solve least squares problem
        A = lstsq(Xx2 @ Xx , Xx2)[0]
        L[i,:] = A[0,:]

    # Functional approximation
    FX = L @ Y

    # leave-one-out cross-validation errors
    RES = np.sqrt(np.mean((Y - FX)*(Y - FX))) / np.std(Y)
    return FX, RES


def compute_res(V, eps_scale=5):
    # Compute cross-validation error and residuals according to reference

    n = V.shape[1]
    RES = np.zeros(n)
    RES[0] = 1
    for i in range(1, n):
        _, RES[i] = local_linear_regression(V[:, i], V[:, :i], eps_scale)

    # Sort eigvec indices to obtain pars. DM rep.
    indices = np.argsort(RES)
    indices = indices[::-1]
    logging.basicConfig(level=logging.INFO)
    #logging.info("\tInitial order or residuals is \n\t%s ", str(RES).strip('[]'))
    #logging.info("\tSorted residuals are \n\t%s ", str(RES[indices]).strip('[]'))
    logging.info("\tSorted indices are %s ", str(indices).strip('[]'))
    return np.squeeze(RES), indices
