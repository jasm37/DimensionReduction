import numpy as np
import logging

from scipy.spatial import distance
from scipy.sparse.linalg import eigsh
from pars_rep_dm import compute_res
import time as tm


# TODO: add MORE references/bibliography
class DiffusionMap:
    '''
    Diffusion Maps class to compute compact representations of data
    In particular, data on manifolds
    TODO: missing exceptions and data validation
    '''
    def __init__(self, step=1, eps=0):
        self.eps = eps
        self.step = step
        self.sq_distance = []
        self.K = []

        # Data array
        self.data = [] # (N,d) data array: N is number of samples, d dimension

        # Eigvectors and eigvalues of DM kernel decomp
        self.eigvec = []
        self.eigval = []

        # Density vector used for Nystrom interpolation
        self.density = []

        # Array containing projection of data into reduced space
        self.proj_data = []
        # Dimension of the projection space.
        # This changes everytime dim_reduction is called
        self.proj_dim = []

        # Number of neighbourhoods to perform RBF interpolation
        self.nnb = 2

        # Set loggers properties
        format = "%(levelname)s:%(name)s:\t%(message)s"
        logging.basicConfig(level=logging.DEBUG, format=format)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def get_distance_matrix(data):
        return distance.cdist(data, data, 'euclidean')

    def _get_sq_distance(self,x, y):
        return distance.cdist(x, y, 'sqeuclidean')

    def compute_ker_eig(self, ker, ndim):
        w, x = eigsh(ker, k=ndim, which='LM', ncv=None)
        return w, x

    def get_eps_from_data(self):
        '''Computes epsilon parameter from distance matrix.
        There are many ways to compute this, here only median
        is implemented. Lafons suggestion is not recommended
        '''
        # Median of the distances:
        self.eps = np.median(np.sqrt(self.sq_distance))

    def get_kernel(self, eps):
        '''Computes the basic kernel matrix(without normalizations)
        TODO: use scikit to complete the distances(may be faster)
        '''
        D = self.sq_distance
        return np.exp(-D / (eps*eps))

    def get_partial_kernel(self, data, eps, nbhd_param):
        '''Computes kernel matrix based on eps or k-nbhd
        TODO implement separate knn and epsilon-nbhd function to create the adjacency matrix
        TODO compare current computation of knn with NearestNeighbourhood from scikitlearn
        :param data: high dimensional data
        :param eps: corresponds to diffusion maps
        :param nbhd_param:  dictionary that contains 'eps' parameter for eps-nbhds
                            or 'k' for k-nearest nbhd
        '''
        default_k = 10 # default value of k in case the one in the dictionary is empty
        default_eps = 1.0 # default value of eps in case the one in the dictionary is empty
        num_samples = data.shape[0]
        ker = np.zeros((num_samples,num_samples))
        dist = self.sq_distance
        if 'k' in nbhd_param:
            k = nbhd_param.get('k', default_k)
            for i in range(num_samples):
                temp = np.argsort(dist[i, :])[:k+1] #indices always contain i, so we take k+1
                for j in temp:
                    ker[i, j] = np.exp(-dist[i,j] / (eps*eps))
        elif 'eps' in nbhd_param:
            rad = nbhd_param.get('eps', 1.0)
            for i in range(num_samples):
                indices = [k for k,v in enumerate(dist[i,:]< rad) if v]
                for j in indices:
                    ker[i, j] = np.exp(-dist[i,j] / (eps*eps))
        else:
            raise ValueError("Unindentified nbhd type")

        return ker

    def set_params(self, data, nbhd_param = None):
        '''Sets data and nbhd
        TODO: This method could go in the constructor, which may change most of the demos
        :param data: high dimensional data
        :param nbhd_param:  dictionary that contains 'eps' parameter for eps-nbhds
                            or 'k' for k-nearest nbhd
        '''
        #
        self.data = data
        self.sq_distance = self._get_sq_distance(data, data)
        # if no epsilon is given then compute it wrt the sq ditance(data)
        if not self.eps: self.get_eps_from_data()
        eps = self.eps
        self.logger.info('Diff. maps eps is %f', eps)
        if nbhd_param:
            self.K = self.get_partial_kernel(data, eps, nbhd_param)
        else:
            self.K = self.get_kernel(eps)

    def construct_dm(self, ndim, pars=False):
        '''Computes eigenvectors and eigenvalues of diffusion operator
        :param ndim: target number of dimensions
        :param pars: Boolean for parsimonious representation: without higher harmonic components
        :return: sets self.eigvec and self.eigval
        '''
        self.proj_dim = ndim
        K = self.K
        alpha = 1

        # Normalization:
        # Approximates geometry without influence of density
        vd = np.sum(K, axis=1)
        self.density = vd
        vd = np.power(vd, -alpha)
        diag = np.diag(vd)
        _K = diag @ K @ diag

        # Compute symmetric kernel (adjoint to the original one):
        # Better computation of eigenvalues and eigenvectors
        vk = np.sum(_K, axis=1)
        # Optional: Create Markov Chain _mc
        # diag_vk = np.diag(1/vk)
        # _mc = diag_vk @ _K
        diag_vec = np.sqrt(vk)
        inv_diag = np.diag(1/diag_vec)
        M = inv_diag @ _K @inv_diag

        # Compute first ndim+1 eigenvalues and discard the first one since it is trivial
        start = tm.time()
        w, x = eigsh(M, k=ndim + 1, which='LM', ncv=None)
        end = tm.time()
        self.logger.info("DM coordinates computation took %f s", end-start)

        # Get decreasing order of evectors and evalues
        w = w[::-1]
        x = x[:, ::-1]

        # Get original eigenvector(instead of the one from the symmetric kernel)
        x = inv_diag @ x
        # Optional: normalize vectors
        norm_x = np.diag(1/np.linalg.norm(x,axis=0))
        x = x @ norm_x

        # first eigenvector/value is trivial
        w = w[1:]
        x = x[:, 1:]

        # Compute parsimonious representation if needed
        if pars:
            start = tm.time()
            # Compute residuals given the eigvectors
            _, indices = compute_res(x)
            w = w[indices]
            x = x[:,indices]
            end = tm.time()
            self.logger.info("Parsimonious rep. of DM took %f s", end-start)

        # Assign to member variables
        self.eigval = w
        self.eigvec = x

    def dim_reduction(self, ndim, pars=False):
        '''Computes low dimensional representation of data
        (N,d) self.data array: N is number of samples, d dimension
        :param ndim: target number of dimensions < d
        :param pars: bool for parsimonious representation: without higher harmonic components
        :return: ndim-dimensional representation of self.data
        '''
        if self.eigvec == [] and self.eigval == []:
            # self.compute_eigdecomp(ndim)
            self.construct_dm(ndim, pars)

        x = self.eigvec
        w = self.eigval

        # Compute data reduction according to Coifman and Lafon :
        # http://www.sciencedirect.com/science/article/pii/S1063520306000546
        self.proj_data = (w[0:ndim]**self.step) * x[:, 0:ndim]
        return w, self.proj_data

    def dm_basis(self, n_components, pars=False):
        '''Returns eigvector and eigvalues of dm decomposition
        '''
        if self.eigvec == [] and self.eigval == []:
            # self.compute_eigdecomp(n_components)
            self.construct_dm(n_components, pars)

        x = self.eigvec
        w = self.eigval

        # Compute data reduction according to Coifman and Lafon :
        # http://www.sciencedirect.com/science/article/pii/S1063520306000546
        self.proj_data = (w[0:n_components] ** self.step) * x[:, 0:n_components]
        return self.eigval[:n_components], self.eigvec[:,:n_components]

    def param_from_indices(self, indices, ndim, pars=True):
        '''Given an array of indices and decreasingly ordered array of eigvectors,
        computes the dimension reduction using the selected eigvectors form the index array
        '''
        indices = np.asarray(indices)
        indices = np.squeeze(indices)

        if self.eigvec == [] and self.eigval == []:
            self.construct_dm(ndim, pars)

        w, x = self.eigval[indices], self.eigvec[:,indices]
        y = (w[0:ndim] ** self.step) * x[:, 0:ndim]
        return y