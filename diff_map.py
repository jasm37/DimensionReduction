import numpy as np
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

#TODO: add references/bibliography
class DiffusionMap:
    def __init__(self, step=1, eps=0):
        self.eps = eps
        self.step = step
        self.sq_distance = 0
        self.K = 0
        self.data = 0

    @staticmethod
    def get_distance_matrix(data):
        return distance.cdist(data, data, 'euclidean')

    @staticmethod
    def get_sq_distance(data):
        return distance.cdist(data, data, 'sqeuclidean')

    def get_eps_from_weight(self):
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

    def get_kernel(self, data, eps):
        # TODO: use scikit to complete the distances(may be faster)
        #D = distance.cdist(data,data,'euclidean')
        D = self.sq_distance
        return np.exp(-D / eps)

    def get_partial_kernel(self, data, eps, nbhd_param):
        # TODO implement separate knn and epsilon-nbhd function to create the adjacency matrix
        # TODO compare current computation of knn with NearestNeighbourhood from scikitlearn
        # Computes kernel matrix based on eps or k-nbhd
        # 'eps' corresponds to diffusion maps
        # 'epsilon' corresponds the nbhd radius
        num_samples = data.shape[0]
        ker = np.zeros((num_samples,num_samples))
        #dist = self.get_distance_matrix(data)
        dist = self.sq_distance
        if 'k' in nbhd_param:
            k = nbhd_param.get('k', 10)
            for i in xrange(num_samples):
                temp = np.argsort(dist[i, :])[:k+1] #indices always contain i, so we take k+1
                for j in temp:
                    ker[i, j] = np.exp(-dist[i,j] / eps)
        elif 'eps' in nbhd_param:
            epsilon = nbhd_param.get('eps', 1.0)
            for i in xrange(num_samples):
                indices = [k for k,v in enumerate(dist[i,:]< epsilon) if v]
                for j in indices:
                    ker[i, j] = np.exp(-dist[i,j] / eps)
        else:
            raise ValueError("Unindentified nbhd type")

        return ker

    def set_params(self, data, nbhd_param = None):
        self.data = data
        self.sq_distance = self.get_sq_distance(data)
        # if no epsilon is given compute it wrt the sq ditance(data)
        if not self.eps: self.get_eps_from_weight()
        eps = self.eps
        print 'Diff. maps eps is ', eps
        if nbhd_param:
            self.K = self.get_partial_kernel(data, eps, nbhd_param)
        else:
            self.K = self.get_kernel(data, eps)

    def get_diffusion_map(self):
        K = self.K
        # W is the kernel/weight matrix
        # Approximates Laplace Beltrami operator: alpha = 1 in Lafon's paper
        vd = np.sum(K, axis=1)
        mult = np.outer(vd,vd)
        _K = K/mult
        vd = np.sum(_K, axis=1)
        mult = np.sqrt(np.outer(vd,vd))
        M = _K/mult

        #This M is a markov chain
        #M = (W.T / W.sum(axis=1)).T
        return M

    def dim_reduction(self, ndim):
        # INPUT:    (N,d) data array: N is number of samples, d dimension
        #           ndim:   desired dimension (ndim<d)
        #           param: contains info about nbhd type for adjancency matrix
        #               knn or eps-nbhd
        # OUTPUT:   first 'ndim' nontrivial eigenvalues and (N,ndim) array
        #           with reduced dimension

        H = self.get_diffusion_map()
        # We get the first ndim+1 eigenvalues and discard the first one since it is 1
        w, x = eigsh(H, k=ndim+1 , which='LM', ncv=None)
        # Get decreasing order of evectors and evalues
        w = w[::-1]
        x = x[:,::-1]
        # Compute data reduction according to Coifman and Lafon :
        # http://www.sciencedirect.com/science/article/pii/S1063520306000546
        # Note: take real part in case of any complex eigenvector/value
        x = (w[1:ndim+1]**self.step) * x[:, 1:ndim+1]
        return w, x


    def get_distance_matrix_and_k(self, data, k):
        print 'Using get_distance_matrix_and_k for testing only'
        # instead of using euclidean, sqeuclidean may be slightly faster
        D = distance.cdist(data,data, 'sqeuclidean')
        k_nbhd = NearestNeighbors(n_neighbors=k )
        k_nbhd.fit(data)
        distances, indices = k_nbhd.kneighbors()
        #change distance to double/float
        return D, indices, distances