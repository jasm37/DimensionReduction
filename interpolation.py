# Contains all methods used to interpolate points from ambient space to DM space and viceversa
import numpy as np
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import scipy.spatial.distance as dist
import scipy.sparse.linalg as ssl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nystrom_ext(x, data, eps, eigval, eigvec):
    # Nyström extension for outlier points.
    # If the original space is Rn and projection space Rd,
    # then this function maps outliers of the data from Rn to Rd
    # similar to the DMAP dim. reduction on the data
    x_vec = x.reshape(1, x.shape[0])
    #d_vec = self._get_sq_distance(x_vec, data)
    d_vec = dist.cdist(x_vec, data, 'sqeuclidean')
    w_vec = np.exp(-d_vec / (eps*eps))
    sum_w = np.sum(w_vec)
    k_vec = w_vec / sum_w
    proj_point = eigvec.T@ k_vec.T
    proj_point = (proj_point.T / eigval).T
    #diag_mat = np.diag(1/eigval)
    #proj_point = diag_mat @ proj_point
    return np.squeeze(proj_point)

def geom_harmonics(x, data, fdata, eps, neig):
    n_elems = fdata.shape[1]
    n_fdim = fdata.shape[0]
    x_vec = x.reshape(1, x.shape[0])
    dist_vec = dist.cdist(x_vec, data, 'sqeuclidean')
    dist_matrix = dist.cdist(data, data, 'sqeuclidean')
    ker_matrix = np.exp(-dist_matrix / eps)
    ker_vec = np.exp(-dist_vec / eps)
    eigval, eigvec = ssl.eigsh(ker_matrix, k=neig, which='LM', ncv=None)
    eigval, eigvec = eigval[::-1], eigvec[:,::-1]
    ext_eigvec = ker_vec @ eigvec / eigval
    ext_array = np.zeros(n_elems)
    proj_farray = np.zeros((n_elems,n_fdim))
    s_list = range(neig)#### s_list might change for multiscale methods
    for i in range(n_elems):
        proj_fdata = 0
        ext_fdata = 0
        for j in s_list:
            eigv = eigvec[:, j]
            proj_coeff = fdata[:, i] @ eigv
            proj_fdata += proj_coeff * eigv
            ext_fdata += proj_coeff * ext_eigvec[:,j]
        ext_array[i] = ext_fdata
        proj_farray[i,:] = proj_fdata

    return ext_array, proj_farray


def rbf_interpolate(x, data, fdata, nnbhd=20):
    # Interpolates from proj./diff. space to orig. space
    # Original space : Rn, projection space: Rd, where d<n
    # Inputs :
    #   x   :   point in Rd
    #   data:   data in Rd
    #   fdata:  image of data in Rn( f(data) )
    #   nnbhd:  number of nbhd points to use for interpolation
    # Output:
    #   interpolated point in Rn and indices of closest points for interpolation

    # compute distances between data and point
    x_vec = x.reshape(1, x.shape[0])
    print("x_vec is ", x_vec.shape, ", data is ", data.shape)
    dist_x = dist.cdist(x_vec, data, 'sqeuclidean')
    # Sorts distances
    indices = np.squeeze(np.argsort(dist_x))[::-1]
    indices = indices[:nnbhd]
    print("temp.shape is ", indices.shape)
    interp_y = fdata[indices, :]
    interp_x = data[indices, :]
    pred = []
    local_eps = np.median(dist_x[:nnbhd])
    # local_eps = 4
    # For each dimension in the output perform radius basis function(gaussian) interpolation
    for j in range(interp_y.shape[1]):
        # additionally, an eps parameter can be defined for the gaussian distribution
        # For more details check Rbf docu in numpy python
        rbfi = Rbf(*interp_x.T, interp_y[:, j], function="linear")
        tt = rbfi(interp_x.T)
        #rbfi = InterpolatedUnivariateSpline(*interp_x.T, interp_y[:, j])
        pred.append(rbfi(*x_vec.T))
    return np.asarray(pred), indices


def kernel(D, eps):
    #return np.exp(-D*D/(eps*eps))
    return np.power(D, eps)


def poly_rbf(x, data, fdata, power=2, nnbhd=2):
    x_vec = x.reshape(1, x.shape[0])
    norm_vec = np.squeeze(dist.cdist(x_vec, data))
    indices = np.squeeze(np.argsort(norm_vec))#[::-1]
    indices = indices[:nnbhd]
    sorted_data = data[indices, :]
    sorted_fdata = fdata[indices,:]
    sorted_dist = dist.squareform(dist.pdist(sorted_data))
    eps = np.mean(sorted_dist)
    alpha = kernel(sorted_dist, eps)
    #alpha = kernel(sorted_dist, power)
    sorted_norm = norm_vec[indices]
    sorted_kern = kernel(sorted_norm, eps)
    #sorted_kern = kernel(sorted_norm, power)
    coeffs = np.linalg.solve(alpha, sorted_fdata)
    interp = coeffs.T @ sorted_kern
    return interp, indices

def onedim_test():
    n_samples = 1000
    n = int(np.sqrt(n_samples))
    x = np.linspace(-10.0, 10.0, n)
    y = np.linspace(-20.0, 20.0, n)
    X, Y = np.meshgrid(x, y)
    #Z = 2 * X + Y
    Z = X*X*X + Y*Y
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    XY = np.vstack((X, Y))
    Z = Z.reshape((Z.shape[0], 1))
    topred = np.array([[0], [0]])
    # pred = rbf_interpolate(topred, XY.T, Z, nnbhd=10)
    pred = poly_rbf(topred, XY.T, Z, power=5, nnbhd=10)
    # print(pred)
    xx = np.linspace(-5.0, 5.0, n)
    yy = np.linspace(-10.0, 10.0, n)
    XX, YY = np.meshgrid(xx, yy)
    XX = XX.reshape(-1)
    YY = YY.reshape(-1)
    topred = np.vstack((XX, YY))
    pred_list = []
    for i in range(XX.shape[0]):
        zz = topred[:, i]
        # zz = zz.reshape(1, zz.shape[0])
        pred, _ = poly_rbf(zz, XY.T, Z, power=2, nnbhd=50)
        pred_list.append(pred)
    pred_arr = np.squeeze(np.asarray(pred_list))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XX, YY, pred_arr, c="blue", cmap=plt.cm.Spectral)
    ax.scatter(X, Y, Z, c="red", cmap=plt.cm.Spectral)
    plt.show()


def vectorfield_test():
    n_samples = 10
    n = int(np.sqrt(n_samples))
    #X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
    X, Y = 2*np.pi*np.random.rand(n_samples), 2*np.pi*np.random.rand(n_samples)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    XY = np.vstack((X, Y))
    U = X*X - Y*Y*X
    V = Y*np.sin(X-Y)
    UV = np.vstack((U, V))


    xx = np.linspace(np.pi/2, 3*np.pi/2, n)
    yy = np.linspace(np.pi/2, 3*np.pi/2, n)
    XX, YY = np.meshgrid(xx, yy)
    XX = XX.reshape(-1)
    YY = YY.reshape(-1)
    topred = np.vstack((XX, YY))
    pred_list = []
    for i in range(XX.shape[0]):
        zz = topred[:, i]
        # zz = zz.reshape(1, zz.shape[0])
        pred, _ = poly_rbf(zz, XY.T, UV.T, power=2, nnbhd=50)
        pred_list.append(pred)
    pred_arr = np.asarray(pred_list)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(X, Y, U, c="blue", cmap=plt.cm.Spectral)
    ax.scatter(XX, YY, pred_arr[:,0], c="red", cmap=plt.cm.Spectral)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(X, Y, V, c="blue", cmap=plt.cm.Spectral)
    ax.scatter(XX, YY, pred_arr[:, 1], c="red", cmap=plt.cm.Spectral)
    plt.show()


if __name__ == "__main__":
    #onedim_test()
    vectorfield_test()





