# Contains all methods used to interpolate points from ambient space to DM space and viceversa
import numpy as np
import logging
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import scipy.spatial.distance as dist
import scipy.sparse.linalg as ssl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GeometricHarmonics():
    # References:
    #   -Main paper:
    #       -Geometric harmonics: A novel tool for multiscale out-of-sample extension of empirical functions
    #           https://www.sciencedirect.com/science/article/pii/S1063520306000522
    #   -Other helpful sources:
    #       -Data Fusion and Multicue Data Matching by Diffusion Maps
    #           http://ieeexplore.ieee.org/document/1704834/
    #       -Reduced Models in Chemical Kinetics via Nonlinear Data-Mining
    #           https://arxiv.org/pdf/1307.6849.pdf

    def __init__(self, data, fdata, eps, neig):
        self.eps = eps
        self.neig = neig
        # data(x) is domain variables, and fdata(f(x)) is the resp. image
        self.data = data

        #Number of dimensions of f(x)
        if fdata.ndim == 1:
            self.n_fdim = 1
            self.fdata = fdata.reshape(fdata.shape[0],1)
        else:
            self.n_fdim = fdata.shape[1]
            self.fdata = fdata
        self.n_elems = fdata.shape[0]
        self.dist_mat = dist.cdist(data, data, 'sqeuclidean')
        # Maximum number of iterations for multiscale method
        self.max_count = 50

        self.fro_error = 0
        self.eigval = []
        self.eigvec = []
        self.ker_matrix = []
        self.proj_fdata = []
        self.proj_coeffs = []

        format = "%(levelname)s:%(name)s:\t%(message)s"
        logfile = "GeomHarm.log"
        # Write log to file
        logging.basicConfig(filename=logfile, level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        # Show log
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(format))
        logging.getLogger('').addHandler(console)

    def load_cached_mat(self, eigvec, eigval, proj_fdata, proj_coeffs, eps):
        assert eigval.shape[0] == self.neig and eigvec.shape[1] == self.neig, \
            "GH:: Number of eigenvalues/vectors is not correct for loaded array"
        assert proj_coeffs.shape[1] == self.n_fdim, \
            "GH:: Dimension of projection matrix is not correct"
        self.eps = eps
        self.eigvec = eigvec
        self.eigval = eigval
        self.proj_fdata = proj_fdata
        self.proj_coeffs = proj_coeffs

    def _compute_eigv(self, ker_mat, neig):
        eigval, eigvec = ssl.eigsh(ker_mat, k=neig, which='LM', ncv=ker_mat.shape[0]-1)
        self.eigval, self.eigvec = eigval[::-1], eigvec[:, ::-1]
        return self.eigval, self.eigvec

    def fit(self):
        self.ker_matrix = np.exp(-self.dist_mat / self.eps)
        self.eigval, self.eigvec = self._compute_eigv(self.ker_matrix, self.neig)
        self.proj_coeffs = (self.fdata.T @ self.eigvec).T
        self.proj_coeffs = np.reshape(self.proj_coeffs, (self.proj_coeffs.shape[0], self.n_fdim))
        proj_farray = np.zeros((self.n_fdim, self.n_elems))
        for i in range(self.n_fdim):
            proj_farray[i,:] = np.sum(self.eigvec*self.proj_coeffs[:,i], axis=1)
        self.proj_fdata = proj_farray
        self.fro_error = np.linalg.norm(self.proj_fdata.T-self.fdata)

    def interpolate(self, x, eps=None):
        # If no eps is given then use self.eps
        #eps = eps if eps is not None else self.eps
        if eps is None:
            eps = self.eps
        x_vec = x.reshape(1, x.shape[0])
        dist_vec = dist.cdist(x_vec, self.data, 'sqeuclidean')
        ker_vec = np.exp(-dist_vec / eps)
        # Compute extension of eigvectors
        ext_eigvec = ker_vec @ self.eigvec / self.eigval
        # Extension array for values to be interpolated
        ext_array = np.zeros(self.n_fdim)
        for i in range(self.n_fdim):
            ext_array[i] = np.sum(self.proj_coeffs[:,i]*ext_eigvec, axis=1)

        return ext_array#, self.proj_fdata

    def mult_interpolate(self, *args, eps=None):
        # If no eps is given then use self.eps
        eps = eps if eps is not None else self.eps
        x_array = np.squeeze(np.asarray(args))
        num_interpolands = x_array.shape[0]
        #x_vec = x.reshape(1, x.shape[0])
        x_vec = x_array

        dist_vec = dist.cdist(x_vec, self.data, 'sqeuclidean')
        ker_vec = np.exp(-dist_vec / eps)
        # Compute extension of eigvectors
        ext_eigvec = ker_vec @ self.eigvec / self.eigval
        # Extension array for values to be interpolated
        ext_array = np.zeros((num_interpolands, self.n_fdim))
        for i in range(self.n_fdim):
            ext_array[:,i] = np.sum(self.proj_coeffs[:, i] * ext_eigvec, axis=1)

        return ext_array#, self.proj_fdata

    def multiscale_fit(self, error):
        if self.fro_error == 0:
            self.fit()

        count = 1
        eps_list = [self.eps]
        error_list = [self.fro_error]

        while self.fro_error > error and count < self.max_count:
            count += 1
            self.eps /= 2
            self.fit()
            eps_list.append(self.eps)
            error_list.append(self.fro_error)

        if count == self.max_count:
            pos = np.argmin(error_list)
            self.eps = eps_list[pos]
            self.fit()

        #self.logger.info("Multiscale fit stopped after %d out of %d iterations, eps %f and error %f ", count, self.max_count, self.eps, self.fro_error)


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
    n_fdim = fdata.shape[1]
    n_elems = fdata.shape[0]
    x_vec = x.reshape(1, x.shape[0])
    dist_vec = dist.cdist(x_vec, data, 'sqeuclidean')
    dist_matrix = dist.cdist(data, data, 'sqeuclidean')
    ker_matrix = np.exp(-dist_matrix / eps)
    ker_vec = np.exp(-dist_vec / eps)
    eigval, eigvec = ssl.eigsh(ker_matrix, k=neig, which='LM', ncv=None)
    eigval, eigvec = eigval[::-1], eigvec[:,::-1]
    ext_eigvec = ker_vec @ eigvec / eigval
    ext_array = np.zeros(n_elems)
    proj_farray = np.zeros((n_fdim,n_elems))
    s_list = range(neig)#### s_list might change for multiscale methods
    for i in range(n_fdim):
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


def multiscale_gm(x, data, fdata, eps, neig ,error):
    var_eps = eps
    proj_error = error + 1
    print("Error is ", error)
    while proj_error > error:
        sol_arr, proj_fdata = geom_harmonics(x, data, fdata, var_eps, neig)
        proj_error = np.linalg.norm(proj_fdata.T-fdata)
        var_eps /= 2
        print("Eps is : ", var_eps, ", proj_error is ", proj_error, ", maximum error is ", np.max(proj_fdata.T-fdata) )
    return sol_arr, proj_fdata


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


def plot_circle():
    x = np.linspace(0,2*np.pi,100)
    Z = np.cos(2*x)
    X = np.cos(x)
    Y = np.sin(x)
    XY = np.vstack((X, Y)).T
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(X,Y,Z)
    plt.show()
    '''

    neig = 70
    gm_error = 0.1
    eps = 1
    gh = GeometricHarmonics(XY, Z, eps=eps, neig=neig)
    gh.multiscale_fit(gm_error)
    nsamples = 100
    xx = np.linspace(-4.0, 4.0, nsamples)
    yy = np.linspace(-4.0, 4.0, nsamples)
    XX, YY = np.meshgrid(xx, yy)
    XX_ = XX.reshape(-1)
    YY_ = YY.reshape(-1)
    data = np.vstack((XX_,YY_)).T
    data = np.ndarray.tolist(data)

    mult_val = gh.mult_interpolate(data)
    mult_val = np.squeeze(np.asarray(mult_val))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(X, Y, Z)
    mult_val = mult_val.reshape((nsamples, nsamples), order='F').T
    ax.plot_surface(XX, YY, mult_val, cmap="autumn",antialiased=True)
    plt.show()


if __name__ == "__main__":
    #onedim_test()
    #vectorfield_test()
    plot_circle()





