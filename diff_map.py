import numpy as np
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

class DiffusionMap:
    def __init__(self, eps, step):
        self.eps = eps
        self.step = step

    @staticmethod
    def get_distance_matrix(data):
        return distance.cdist(data, data, 'euclidean')

    def get_kernel(self, data, eps):
        # TODO: use scikit to complete the distances(may be faster)
        D = distance.cdist(data,data,'euclidean')
        return np.exp(-D ** 2 / eps)

    def get_partial_kernel(self, data, delta, param):
        # TODO implement separate knn and epsilon-nbhd function to create the adjacency matrix
        # TODO compare current computation of knn with NearestNeighbourhood from scikitlearn
        # Computes kernel matrix based on eps or k-nbhd
        # delta is the epsilon of diffusion maps
        num_samples = data.shape[0]
        ker = np.zeros((num_samples,num_samples))
        dist = self.get_distance_matrix(data)

        if 'k' in param:
            k = param.get('k', 10)
            for i in xrange(num_samples):
                temp = np.argsort(dist[i, :])[:k+1] #indices always contain i, so we take k+1
                for j in temp:
                    ker[i, j] = np.exp(-dist[i,j] ** 2 / delta)
        elif 'eps' in param:
            eps = param.get('eps', 1.0)
            for i in xrange(num_samples):
                indices = [k for k,v in enumerate(dist[i,:]< eps) if v]
                for j in indices:
                    ker[i, j] = np.exp(-dist[i,j] ** 2 / delta)
        else:
            raise ValueError("Unindentified nbhd type")

        return ker

    def get_diffusion_map(self, data, eps, param = None):
        # Method could also be named "get_diffusion_map"
        if param:
            W = self.get_partial_kernel(data, eps, param)
        else:
            W = self.get_kernel(data, eps)
        # W is the kernel/weight matrix
        # Approximates Laplace Beltrami operator: alpha = 1 in Lafon's paper
        vd = np.sum(W, axis=1)
        mult = np.outer(vd,vd)
        W = W/mult
        vd = np.sum(W, axis=1)
        mult = np.sqrt(np.outer(vd,vd))
        M = W/mult

        #This M is a markov chain
        #M = (W.T / W.sum(axis=1)).T
        return M

    def dim_reduction(self, data, ndim, param = None):
        # INPUT:    (N,d) data array: N is number of samples, d dimension
        #           ndim:   desired dimension (ndim<d)
        #           param: contains info about nbhd type for adjancency matrix
        #               knn or eps-nbhd
        # OUTPUT:   first 'ndim' nontrivial eigenvalues and (N,ndim) array
        #           with reduced dimension

        H = self.get_diffusion_map(data, self.eps, param)
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