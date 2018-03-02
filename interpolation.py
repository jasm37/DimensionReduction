# Contains all methods used to interpolate points from ambient space to DM space and viceversa
import numpy as np
import logging
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import scipy.spatial.distance as dist
import scipy.sparse.linalg as ssl
#import diff_map as diffmap
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

    def __init__(self, data, fdata, eps, neig, delta=0.0):
        self.eps = eps
        self.neig = neig
        self.delta = delta
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
        self.max_count = 30

        self.fro_error = 0
        self.fro_error_list = np.full_like(fdata[0], 0)
        self.eigval_list = []
        self.eigvec_list = []
        self.proj_coeffs_list = []
        self.eps_vec = [self.eps]*fdata.shape[1]

        self.eigval = []
        self.eigvec = []
        self.ker_matrix = []
        self.proj_fdata = np.full_like(fdata, 0)
        self.proj_coeffs = np.full_like(fdata[0], 0)

        format = "%(levelname)s:%(name)s:\t%(message)s"
        logging.basicConfig(level=logging.DEBUG, format=format)
        self.logger = logging.getLogger(__name__)

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
        eigval, eigvec = ssl.eigsh(ker_mat, k=neig, which='LM', ncv=ker_mat.shape[0]-1, maxiter=5000, tol=1E-6)
        self.eigval, self.eigvec = eigval[::-1], eigvec[:, ::-1]
        if self.delta != 0.0:
            delta_mask = self.eigval > self.delta*self.eigval[0]
            self.eigval = self.eigval[delta_mask]
            self.eigvec = self.eigvec[:,delta_mask]
        return self.eigval, self.eigvec

    def _iterative_fit(self):
        """Fit per variable
            Performs fit for all variables and stores results in the respective lists
            to be analysed in multiscale_var_fit
        """
        self.ker_matrix = np.exp(-self.dist_mat / self.eps)
        eigval, eigvec = self._compute_eigv(self.ker_matrix, self.neig)
        proj_coeffs = (self.fdata.T @ eigvec).T
        proj_coeffs = np.reshape(proj_coeffs, (proj_coeffs.shape[0], self.n_fdim))
        proj_farray = np.zeros((self.n_fdim, self.n_elems))
        for i in range(self.n_fdim):
            proj_farray[i,:] = np.sum(eigvec*proj_coeffs[:,i], axis=1)
        self.fro_error_list = np.linalg.norm(proj_farray.T-self.fdata, axis=0)
        self.eigvec_list.append(eigvec)
        self.eigval_list.append(eigval)

    def _iterative_interpolate(self, *args):
        """Performs interpolation per variable
        """
        # TODO 1: Can only be called if mult_var_fir was used first(not multiscale_fit)
        # TODO 2: Vectorize(using tensors/3d arrays) instead of using for loop(only if possible)
        x_vec = np.squeeze(np.asarray(args))
        n_interpolands = x_vec.shape[0]
        one_comp = False
        if x_vec.ndim == 1:
            one_comp = True
            x_vec = x_vec.reshape(1, n_interpolands)
        dist_vec = dist.cdist(x_vec, self.data, 'sqeuclidean')
        ext_array = np.zeros((n_interpolands, self.n_fdim))
        for i in range(self.n_fdim):
            loc_eigvec = self.eigvec_list[i]
            loc_eigval = self.eigval_list[i]
            loc_f = self.fdata[:, i]
            loc_pf = np.inner(loc_f, loc_eigvec.T)
            self.proj_fdata[:, i] = loc_eigvec @ loc_pf
            ker_mat = np.exp(-dist_vec / self.eps_vec[i])
            loc_phi = (1/loc_eigval) * (ker_mat@loc_eigvec)
            loc_approx = loc_pf @loc_phi.T
            ext_array[:, i] = loc_approx
        # TODO 3: (Bug)If only one point is interpolated then the result should be 1-dimensional
        if one_comp:
            ext_array = ext_array[0,:]
        return ext_array

    def fit(self):
        self.ker_matrix = np.exp(-self.dist_mat / self.eps)
        self.eigval, self.eigvec = self._compute_eigv(self.ker_matrix, self.neig)
        self.proj_coeffs = (self.fdata.T @ self.eigvec).T
        self.proj_coeffs = np.reshape(self.proj_coeffs, (self.proj_coeffs.shape[0], self.n_fdim))
        proj_farray = np.zeros((self.n_fdim, self.n_elems))
        for i in range(self.n_fdim):
            proj_farray[i, :] = np.sum(self.eigvec * self.proj_coeffs[:, i], axis=1)
        self.proj_fdata = proj_farray.T
        self.fro_error = np.linalg.norm(self.proj_fdata.T - self.fdata)

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
        if self.fro_error == 0.0:
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

    def multiscale_var_fit(self, error):
        # pos stores what eigendecomposition results in the smallest residue per component
        pos = [0] * self.n_fdim
        if all(ferror == 0 for ferror in self.fro_error_list):
            self._iterative_fit()

        count = 1
        eps_list = [self.eps]
        error_list = [self.fro_error_list]
        current_err = self.fro_error_list

        # If one residual is bigger than error then continue process.
        # Residuals will be computed for all components
        # TODO 1: Use some data structure to make this faster.
        #       Currently storing all results(eigendecompositions and epsilons)
        while max(current_err) > error and count < self.max_count:
            count += 1
            self.eps /= 2
            self._iterative_fit()
            eps_list.append(self.eps)
            # TODO 2: Use np.argmin command for lists instead of converting to array
            error_list.append(self.fro_error_list)
            error_arr = np.asarray(error_list)
            pos = np.argmin(error_arr, axis=0)
            current_err = error_arr[pos, np.arange(len(pos))]

        # Assign selected eigenvalues/vectors and epsilons for weight matrices
        self.f_scale_args = pos
        self.eps_vec = np.asarray([eps_list[ind] for ind in pos])
        self.eigval_list = [self.eigval_list[ind] for ind in pos]
        self.eigvec_list = [self.eigvec_list[ind] for ind in pos]

        # Print eps and residuals
        '''
        max_pos = np.argmax(self.eps_vec)
        min_pos = np.argmin(self.eps_vec)
        max_err = np.max(current_err)
        min_err = np.min(current_err)
        self.logger.info('Max and min pos and values (%d, %f) and (%d, %f)'
                         % (max_pos, self.eps_vec[max_pos], min_pos, self.eps_vec[min_pos]))
        self.logger.info('Max and min errors %f, %f' % (max_err, min_err ))
        '''

def nnbhd_mean(*args, data, fdata, nnbhd=10):
    x_array = np.squeeze(np.asarray(args))
    dist_mat = dist.cdist(x_array, data)
    out = np.zeros((x_array.shape[0], fdata.shape[1]))
    for i in range(x_array.shape[0]):
        ind = np.argsort(dist_mat[i, :])[:nnbhd]
        local_dist = dist_mat[i, ind]
        local_fdata = fdata[ind]
        #num = np.dot(local_dist, local_fdata)
        #den = np.sum(local_dist)
        out[i, :] = np.mean(local_fdata, axis=0)
    return out


def inv_weight(*args, data, fdata, nnbhd=10):
    x_array = np.squeeze(np.asarray(args))
    dist_arr = dist.cdist(x_array, data)
    inv_dist = 1 / dist_arr
    out = np.zeros((x_array.shape[0], fdata.shape[1]))
    for i in range(x_array.shape[0]):
        #arr = arr
        ind = np.argsort(inv_dist[i,:])[::-1][:nnbhd]
        local_dist = inv_dist[i,ind]
        local_fdata = fdata[ind]
        num = np.dot(local_dist, local_fdata)
        den = np.sum(local_dist)
        out[i,:] = num/den

    #nums = inv_dist @ fdata
    #den = np.sum(inv_dist, axis=1)

    #for i in range(len(args)):
    #    y = num / den
    #y = nums.T / den
    #print(np.max(y), np.min(y))
    return out


def nystrom_extension(x, data, dm):
    eps = dm.eps
    eigvec = dm.eigvec[:,:2]
    eigval = dm.eigval[:2]
    density = dm.density
    x_vec = x.reshape(1, x.shape[0])
    d_vec = dist.cdist(x_vec, data, 'sqeuclidean')
    w_vec = np.exp(-d_vec / (eps*eps))
    sum_w = np.sum(w_vec)
    k_vec = w_vec / (sum_w * density)
    sum_k = np.sum(k_vec)
    m_vec = k_vec / sum_k
    #k_vec = w_vec / sum_w
    proj_point = eigvec.T@ m_vec.T
    #proj_point = (proj_point.T / eigval).T
    return np.squeeze(proj_point)


def mult_nystrom_extension(x, data, dm, n_comp):
    """Nystrom extension
    Reference :     Diffusion Maps, Reduction Coordinates, and Low Dimensional Representation of Stochastic Systems
                    R. R. Coifman and I. G. Kevrekidis and S. Lafon and M. Maggioni and B. Nadler
                    http://epubs.siam.org/doi/abs/10.1137/070696325
    :param x:   out-of-sample points to be interpolated
    :param data:    points used for interpolation
    :param dm:  Diffusion Maps(DM) class used for data
    :param n_comp: number of components used with DM
    :return:    Extension of out-of-sample points in DM space
    """
    eps = dm.eps
    eigvec = dm.eigvec[:, :n_comp]
    eigval = dm.eigval[:n_comp]
    density = dm.density
    x_vec = np.asarray(x)
    d_vec = dist.cdist(x_vec, data, 'sqeuclidean')
    w_vec = np.exp(-d_vec / (eps * eps))
    sum_w = np.sum(w_vec, axis=1)
    k_vec = ((w_vec / density).T / sum_w).T
    sum_k = np.sum(k_vec, axis=1)
    m_vec = (k_vec.T / sum_k).T
    proj_point = eigvec.T @ m_vec.T
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
    return np.asarray(pred)#, indices


def kernel(D, eps):
    return np.exp(-D*D/(eps*eps))
    #return np.power(D, 3)


def multi_rbf(x_array, data, fdata, power=2, nnbhd=10, eps=1):
    i = 0
    out_arr = np.zeros((len(x_array), fdata.shape[1]))
    x_array = np.asarray(x_array)
    for point in x_array:
        #point = np.asarray(point)
        interp,_ = poly_rbf(point, data, fdata, power, nnbhd, eps=eps)
        out_arr[i,:] = interp
        i+=1

    return out_arr


def poly_rbf(x, data, fdata, power=2, nnbhd=10, eps=1):
    x_vec = x.reshape(1, x.shape[0])
    norm_vec = np.squeeze(dist.cdist(x_vec, data))
    indices = np.squeeze(np.argsort(norm_vec))#[::-1]
    rr = norm_vec[indices]
    indices = indices[:nnbhd]
    sorted_data = data[indices, :]
    sorted_data, indices = np.unique(sorted_data, axis=0, return_index=True)
    sorted_fdata = fdata[indices,:]
    sorted_dist = dist.squareform(dist.pdist(sorted_data))
    eps = np.median(sorted_dist)
    #print()
    alpha = kernel(sorted_dist, eps)
    #alpha = kernel(sorted_dist, power)
    sorted_norm = norm_vec[indices]
    sorted_kern = kernel(sorted_norm, eps)
    #sorted_kern = kernel(sorted_norm, power)
    #det = np.linalg.det(alpha)
    #print("Det is ", det)
    coeffs = np.linalg.solve(alpha, sorted_fdata)
    interp = coeffs.T @ sorted_kern
    return interp, indices


def interpolate_gh_byparts(data, fdata, interpoland, eps, delta, neig, nnbhd, gm_error):
    interpoland = np.asarray(interpoland)
    dist_vec = dist.cdist(interpoland, data, 'sqeuclidean')
    result_array = np.zeros((interpoland.shape[0],fdata.shape[1]))
    for i in range(interpoland.shape[0]):
        temp = np.argsort(dist_vec[i, :])[:nnbhd]
        local_data = data[temp]
        local_fdata = fdata[temp]
        gh = GeometricHarmonics(local_data, local_fdata, eps=eps, neig=neig, delta=delta)
        #gh.multiscale_fit(gm_error)
        gh.multiscale_var_fit(gm_error)
        #result = gh.interpolate(interpoland[i])
        result = gh._iterative_interpolate(interpoland[i])
        #result_list.append(result)
        result_array[i,:] = result
    return result_array


def onedim_test():
    n_samples = 1000
    n = int(np.sqrt(n_samples))
    x = np.linspace(-10.0, 10.0, n)
    y = np.linspace(-20.0, 20.0, n)
    X, Y = np.meshgrid(x, y)
    #Z = 2 * X + Y
    Z = X*X*X + Y*Y + np.exp((X*Y + X - Y) / 150) + X*Y*np.sin(Y)
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
        pred, _ = poly_rbf(zz, XY.T, Z, power=1, nnbhd=50)
        pred_list.append(pred)
    pred_arr = np.squeeze(np.asarray(pred_list))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XX, YY, pred_arr, c="blue", cmap=plt.cm.Spectral)
    ax.scatter(X, Y, Z, c="red", cmap=plt.cm.Spectral)
    plt.show()


def onedim_test_inv():
    n_samples = 1000
    n = int(np.sqrt(n_samples))
    x = np.linspace(-10.0, 10.0, n)
    y = np.linspace(-20.0, 20.0, n)
    X, Y = np.meshgrid(x, y)
    #Z = 2 * X + Y
    Z = X*X*X + Y*Y + np.exp((X*Y + X - Y) / 150) + X*Y*np.sin(Y)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    XY = np.vstack((X, Y))
    Z = Z.reshape((Z.shape[0], 1))
    topred = np.array([[0], [0]])
    # pred = rbf_interpolate(topred, XY.T, Z, nnbhd=10)
    #pred = poly_rbf(topred, XY.T, Z, power=5, nnbhd=10)
    # print(pred)
    n_samples = 500
    n = int(np.sqrt(n_samples))
    xx = np.linspace(-5.0, 5.0, n)
    yy = np.linspace(-10.0, 10.0, n)
    XX, YY = np.meshgrid(xx, yy)
    XX = XX.reshape(-1)
    YY = YY.reshape(-1)
    topred = np.vstack((XX, YY))
    pred_list = []
    '''
    for i in range(XX.shape[0]):
        zz = topred[:, i]
        # zz = zz.reshape(1, zz.shape[0])
        pred, _ = poly_rbf(zz, XY.T, Z, power=1, nnbhd=50)
        pred_list.append(pred)
    '''
    pred_arr = inv_weight(topred.T, data=XY.T, fdata=Z)
    #pred_arr = np.squeeze(np.asarray(pred_list))
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XX, YY, pred_arr, c="blue", cmap=plt.cm.Spectral)
    ax.scatter(X, Y, Z, c="red", cmap=plt.cm.Spectral)
    plt.show()


def vectorfield_test():
    n_samples = 2000
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
    x = np.linspace(0,2*np.pi,300)
    Z = 10*np.cos(3*x)
    X = np.cos(x)
    Y = np.sin(x)
    XY = np.vstack((X, Y)).T
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(X,Y,Z)
    plt.show()
    '''

    neig = 99#70
    gm_error = 0.001
    eps = 1000000
    gh = GeometricHarmonics(XY, Z, eps=eps, neig=neig)
    gh.multiscale_fit(gm_error)
    print("Frob error is ", gh.fro_error)
    print("Frob eps is ", gh.eps)
    nsamples = 100
    xx = np.linspace(-1.0, 1.0, nsamples)
    yy = np.linspace(-1.0, 1.0, nsamples)
    XX, YY = np.meshgrid(xx, yy)
    XX_ = XX.reshape(-1)
    YY_ = YY.reshape(-1)
    data = np.vstack((XX_,YY_)).T
    data = np.ndarray.tolist(data)
    s_arr = np.array([0.3,0.3])
    sz = gh.interpolate(s_arr)

    mult_val = gh.mult_interpolate(data)
    mult_val = np.squeeze(np.asarray(mult_val))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(X, Y, Z)
    mult_val = mult_val.reshape((nsamples, nsamples), order='F').T
    ax.plot_surface(XX, YY, mult_val, cmap="autumn",antialiased=True)
    ax.scatter(s_arr[0], s_arr[1], sz, marker="x", c="black")
    plt.show()


def plot_inner_circle():
    x = np.linspace(0,2*np.pi,300)

    Z1 = 10*np.cos(5*x)
    #Z1 = np.ones(x.shape)*1.2
    X1 = np.cos(x)
    Y1 = np.sin(x)
    XY1 = np.vstack((X1, Y1)).T

    Z2 = 10 * np.cos(2 * x)
    #Z2 = np.ones(x.shape)*1
    X2 = 0.3*np.cos(x)
    Y2 = 0.3*np.sin(x)
    XY2 = np.vstack((X2, Y2)).T

    XY = np.vstack((XY1, XY2))
    Z = np.hstack((Z1, Z2))
    X = np.hstack((X1, X2))
    Y = np.hstack((Y1, Y2))

    neig = 99#70
    gm_error = 0.001
    eps = 1000#1000000
    # delta << 1
    delta = 0.00001#0.001
    gh = GeometricHarmonics(XY, Z, eps=eps, neig=neig, delta=delta)
    gh.multiscale_fit(gm_error)
    print("Frob error is ", gh.fro_error)
    print("Frob eps is ", gh.eps)
    print("Number of eigvalues is ", len(gh.eigval))
    nsamples = 100

    width = height = 2
    xx = np.linspace(-width, height, nsamples)
    yy = np.linspace(-width, height, nsamples)
    XX, YY = np.meshgrid(xx, yy)
    XX_ = XX.reshape(-1)
    YY_ = YY.reshape(-1)
    data = np.vstack((XX_,YY_)).T
    data = np.ndarray.tolist(data)
    s_arr = np.array([1.0,1.0])
    sz = gh.interpolate(s_arr)

    mult_val = gh.mult_interpolate(data)
    mult_val = np.squeeze(np.asarray(mult_val))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(X, Y, Z)
    plt.plot(X1, Y1, Z1, c="blue")
    plt.plot(X2, Y2, Z2, c="blue")
    mult_val = mult_val.reshape((nsamples, nsamples), order='F').T
    ax.plot_surface(XX, YY, mult_val, cmap="autumn",antialiased=True)
    ax.scatter(s_arr[0], s_arr[1], sz, marker="x", c="black")
    plt.show()


def inv_main():
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

    neig = 5#70
    gm_error = 1
    eps = 1
    #gh = GeometricHarmonics(XY, Z, eps=eps, neig=neig)

    #gh.multiscale_fit(gm_error)
    nsamples = 100
    xx = np.linspace(-4.0, 4.0, nsamples)
    yy = np.linspace(-4.0, 4.0, nsamples)
    XX, YY = np.meshgrid(xx, yy)
    XX_ = XX.reshape(-1)
    YY_ = YY.reshape(-1)
    data = np.vstack((XX_,YY_)).T
    data = np.ndarray.tolist(data)

    #mult_val = gh.mult_interpolate(data)
    mult_val = inv_weight(data, data=XY, fdata=Z)
    mult_val = np.squeeze(np.asarray(mult_val))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(X, Y, Z)
    mult_val = mult_val.reshape((nsamples, nsamples), order='F').T
    map = ax.plot_surface(XX, YY, mult_val, cmap="autumn",antialiased=True)
    fig.colorbar(map)
    plt.show()



if __name__ == "__main__":
    #onedim_test()
    #vectorfield_test()
    #plot_circle()
    plot_inner_circle()
    #inv_main()
    #onedim_test_inv()








