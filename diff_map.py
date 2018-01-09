import numpy as np
import logging
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.interpolate import griddata, Rbf
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from pars_rep_dm import compute_res
import time as tm

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)


#TODO: add MORE references/bibliography
class DiffusionMap:
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

        # Array containing projection of data into reduced space
        self.proj_data = []
        # Dimension of the projection space.
        # This changes everytime dim_reduction is called
        self.proj_dim = []

        # Number of neighbourhoods to perform RBF interpolation
        self.nnb = 2

        # Set loggers properties
        format = "%(levelname)s:%(name)s:\t%(message)s"
        logfile = "DiffMap.log"
        # Write log to file
        logging.basicConfig(filename=logfile, level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        # Show log
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(format))
        logging.getLogger('').addHandler(console)

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
        self.eps = np.sqrt(np.median(self.sq_distance))
        '''
        # According to Lafons dissertation page 33
        # W is squared distance matrix obtained from data
        W = self.sq_distance
        size = W.shape[0]
        v = np.where(W > 0, W, W.max()).min(1)
        self.eps = np.sqrt(np.sum(v)/size)
        '''

    def get_kernel(self, eps):
        # Computes the basic kernel matrix(without normalizations)
        # TODO: use scikit to complete the distances(may be faster)
        D = self.sq_distance
        return np.exp(-D / (eps*eps))

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
        # Sets data and nbhd
        # TODO: This method could go in the constructor, which may change
        #       most of the demos
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

    def get_diffusion_map(self):
        # Computes normalized kernel of DM

        # K is the matrix exp(-d^2/eps^2), where d is the distances matrix
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

    def constructDifMap(self, ndim, pars=False):
        self.proj_dim = ndim
        K = self.K
        alpha = 1

        # Normalization:
        # Approximates geometry without influence of density
        vd = np.sum(K, axis=1)
        vd = np.power(vd, -alpha)
        diag = np.diag(vd)
        _K = diag @ K @ diag

        vk = np.sum(_K, axis=1)
        diag_vk = np.diag(1/vk)
        _K = _K @ diag_vk

        # Compute symmetric kernel (adjoint to the original one):
        # Better computation of eigenvalues and eigenvectors
        vd = np.sum(_K, axis=1)
        vd = 1 / np.sqrt(vd)
        diag = np.diag(vd)
        M = diag @ _K @ diag
        #M = (M + M.T)/2



        # Compute first ndim+1 eigenvalues and discard the first one since it is trivial
        start = tm.time()
        w, x = eigsh(M, k=ndim + 1, which='LM', ncv=None)
        end = tm.time()
        self.logger.info("DM coordinates computation took %f s", end-start)

        # Get decreasing order of evectors and evalues
        w = w[::-1]
        x = x[:, ::-1]

        # Get original eigenvector(instead of the one from the symmetric kernel)
        x = diag @ x
        norm_x = np.diag(1/LA.norm(x,axis=0))
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
        # Input:    (N,d) data array: N is number of samples, d dimension
        #           ndim:   desired dimension (ndim<d)
        #           param: contains info about nbhd type for adjancency matrix
        #               knn or eps-nbhd
        # Output:   first 'ndim' nontrivial eigenvalues and (N,ndim) array
        #           with reduced dimension

        if self.eigvec == [] and self.eigval == []:
            # self.compute_eigdecomp(ndim)
            self.constructDifMap(ndim, pars)

        x = self.eigvec
        w = self.eigval
        # Compute data reduction according to Coifman and Lafon :
        # http://www.sciencedirect.com/science/article/pii/S1063520306000546
        self.proj_data = (w[0:ndim]**self.step) * x[:, 0:ndim]
        return w, self.proj_data

    def dm_basis(self, n_components, pars=False):
        # Input:    (N,d) data array: N is number of samples, d dimension
        #
        # Output:   first 'ndim' nontrivial eigenvalues and (N,ndim) array
        #           with reduced dimension
        if self.eigvec == [] and self.eigval == []:
            # self.compute_eigdecomp(n_components)
            self.constructDifMap(n_components, pars)

        x = self.eigvec
        w = self.eigval

        # Compute data reduction according to Coifman and Lafon :
        # http://www.sciencedirect.com/science/article/pii/S1063520306000546
        self.proj_data = (w[0:n_components] ** self.step) * x[:, 0:n_components]
        return self.eigval[:n_components], self.eigvec[:,:n_components]

    def param_from_indices(self, indices, ndim, pars=True):
        # Given an array of indices and decreasingly ordered array of eigvectors,
        # computes the dimension reduction using the selected eigvectors form the index array
        indices = np.asarray(indices)
        indices = np.squeeze(indices)
        #self.compute_eigdecomp(ndim=10)
        if self.eigvec == [] and self.eigval == []:
            # self.compute_eigdecomp(ndim)
            self.constructDifMap(ndim, pars)
        #self.constructDifMap(ndim)
        w, x = self.eigval[indices], self.eigvec[:,indices]
        y = (w[0:ndim] ** self.step) * x[:, 0:ndim]
        return y

    def permute_indices(self, indices):
        self.eigval = self.eigval[indices]
        self.eigvec = self.eigvec[:,indices]