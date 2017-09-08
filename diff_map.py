import numpy as np
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.interpolate import griddata, Rbf
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

#TODO: add references/bibliography
class DiffusionMap:
    def __init__(self, step=1, eps=0):
        self.eps = eps
        self.step = step
        self.sq_distance = 0
        self.K = None

        # Data array
        self.data = None

        # Eigvectors and eigvalues of DM kernel decomp
        self.eigvec = None
        self.eigval = None

        # Array containing projection of data into reduced space
        self.proj_data = None
        # Dimension of the projection space.
        # This changes everytime dim_reduction is called
        self.proj_dim = None

        # Number of neighborhoods to perform RBF interpolation
        self.nnb = 4


    @staticmethod
    def get_distance_matrix(data):
        return distance.cdist(data, data, 'euclidean')

    def _get_sq_distance(self,x, y):
        return distance.cdist(x, y, 'sqeuclidean')

    def get_eps_from_data(self):
        # Computes epsilon parameter from distance matrix.
        # There are many ways to compute this, here only median
        # and Lafons suggestion are implemented

        # Median of the distances:
        self.eps = np.median(self.sq_distance)
        '''
        # According to Lafons dissertation page 33
        # W is squared distance matrix obtained from data
        W = self.sq_distance
        size = W.shape[0]
        v = np.where(W > 0, W, W.max()).min(1)
        self.eps = np.sum(v)/size
        '''

    def get_kernel(self,  eps):
        # Computes the basic kernel matrix(without normalizations)
        # TODO: use scikit to complete the distances(may be faster)
        D = self.sq_distance
        return np.exp(-D / eps)

    def get_partial_kernel(self, data, eps, nbhd_param):
        # Computes kernel matrix based on eps or k-nbhd
        # 'eps' corresponds to diffusion maps
        # 'rad' corresponds the nbhd radius
        # TODO implement separate knn and epsilon-nbhd function to create the adjacency matrix
        # TODO compare current computation of knn with NearestNeighbourhood from scikitlearn
        default_k = 10 # default value of k in case the one in the dictionary is empty
        default_eps = 1.0 # default value of eps in case the one in the dictionary is empty
        num_samples = data.shape[0]
        ker = np.zeros((num_samples,num_samples))
        #dist = self.get_distance_matrix(data)
        dist = self.sq_distance
        if 'k' in nbhd_param:
            k = nbhd_param.get('k', default_k)
            for i in range(num_samples):
                temp = np.argsort(dist[i, :])[:k+1] #indices always contain i, so we take k+1
                for j in temp:
                    ker[i, j] = np.exp(-dist[i,j] / eps)
        elif 'eps' in nbhd_param:
            rad = nbhd_param.get('eps', 1.0)
            for i in range(num_samples):
                indices = [k for k,v in enumerate(dist[i,:]< rad) if v]
                for j in indices:
                    ker[i, j] = np.exp(-dist[i,j] / eps)
        else:
            raise ValueError("Unindentified nbhd type")

        return ker

    def set_params(self, data, nbhd_param = None):
        # Sets data and nbhd
        # TODO: This method could go in the constructor, which may change
        #       most of the demos
        self.data = data
        self.sq_distance = self._get_sq_distance(data, data)
        # if no epsilon is given then compute it wrt the sq ditance(data)
        if not self.eps: self.get_eps_from_data()
        eps = self.eps
        print('Diff. maps eps is ', eps)
        if nbhd_param:
            self.K = self.get_partial_kernel(data, eps, nbhd_param)
        else:
            self.K = self.get_kernel(eps)

    def get_diffusion_map(self):
        # Computes normalized kernel of DM

        # K is the matrix exp(-d^2/eps), where d is the distances matrix
        K = self.K
        # M is the kernel/weight matrix
        # Approximates Laplace Beltrami operator: alpha = 1 in Lafon's paper

        '''
        vd = np.sum(K, axis=1)
        mult = np.outer(vd,vd)
        _K = K/mult
        vd = np.sum(_K, axis=1)
        mult = np.sqrt(np.outer(vd,vd))
        M = _K/mult
        '''
        # The following method computes same M as the commented above,
        # but is resembles the equations in the papers. The above one
        # might be faster though.

        alpha = 1
        vd = np.sum(K, axis=1)
        vd = np.power(vd, -alpha)
        diag = np.diag(vd)
        _K = diag @ K @ diag
        vd = np.sum(_K, axis=1)
        vd = 1/np.sqrt(vd)
        diag = np.diag(vd)
        M = diag @ _K @ diag
        return M

    def compute_eigdecomp(self, ndim):
        # Given a target dimension, computes the eigvectors and eigvalues
        # of the DM kernel in decreasing order
        H = self.get_diffusion_map()
        self.proj_dim = ndim
        # We get the first ndim+1 eigenvalues and discard the first one since it is 1
        w, x = eigsh(H, k=ndim + 1, which='LM', ncv=None)
        # Get decreasing order of evectors and evalues
        w = w[::-1]
        x = x[:, ::-1]
        # First eigenvalue/vector contains no info so it is ignored
        self.eigval = w[1:]
        self.eigvec = x[:, 1:]

    def dim_reduction(self, ndim):
        # Input:    (N,d) data array: N is number of samples, d dimension
        #           ndim:   desired dimension (ndim<d)
        #           param: contains info about nbhd type for adjancency matrix
        #               knn or eps-nbhd
        # Output:   first 'ndim' nontrivial eigenvalues and (N,ndim) array
        #           with reduced dimension

        if self.eigvec is not None and self.eigval is not None:
            pass
        else:
            self.compute_eigdecomp(ndim)

        x = self.eigvec
        w = self.eigval
        # Compute data reduction according to Coifman and Lafon :
        # http://www.sciencedirect.com/science/article/pii/S1063520306000546
        self.proj_data = y = (w[0:ndim]**self.step) * x[:, 0:ndim]
        return w, y

    def dm_basis(self, n_components):
        # Input:    (N,d) data array: N is number of samples, d dimension
        #
        # Output:   first 'ndim' nontrivial eigenvalues and (N,ndim) array
        #           with reduced dimension
        if self.eigvec is not None and self.eigval is not None:
            pass
        else:
            self.compute_eigdecomp(n_components)
        return self.eigval[:n_components], self.eigvec[:,:n_components]

    def nystrom_ext(self, x):
        # Nyström extension for outlier points.
        # If the original space is Rn and projection space Rd,
        # then this function maps outliers of the data from Rn to Rd
        # similar to the DMAP dim. reduction on the data
        data = self.data
        x_vec = x.reshape(1, x.shape[0])
        d_vec = self._get_sq_distance(x_vec, data)
        w_vec = np.exp(-d_vec / self.eps)
        sum_w = np.sum(w_vec)
        k_vec = w_vec / sum_w
        proj_point = np.matmul(self.eigvec.T,k_vec.T)
        return proj_point

    def param_from_indices(self, indices, ndim):
        # Given an array of indices and decreasingly ordered array of eigvectors,
        # computes the dimension reduction using the selected eigvectors form the index array
        indices = np.asarray(indices)
        indices = np.squeeze(indices)
        self.compute_eigdecomp(ndim=10)
        w, x = self.eigval[indices], self.eigvec[:,indices]
        y = (w[0:ndim] ** self.step) * x[:, 0:ndim]
        return y

    def rbf_interpolate(self, x):
        # Interpolates from proj./diff. space to orig. space
        # Original space : Rn, projection space: Rd, where d<n
        # Input : point in Rd
        # Output: interpolated point in Rn
        k = self.nnb
        # compute distances between data and point
        x_vec = x.reshape(1,x.shape[0])
        dist_x = self._get_sq_distance(x_vec, self.proj_data).reshape(-1)
        # Sorts distances
        temp = np.argsort(dist_x)[:k]
        interp_y = self.data[temp,:]
        interp_x = self.proj_data[temp,:]
        pred = []
        # For each dimension in the output perform radius basis function(gaussian) interpolation
        for j in range(interp_y.shape[1]):
            # additionally, an eps parameter can be defined for the gaussian distribution
            # For more details check Rbf docu in numpy python
            rbfi = Rbf(*interp_x.T, interp_y[:,j], function="gaussian")
            pred.append(rbfi(*x_vec.T))
        return np.asarray(pred), temp