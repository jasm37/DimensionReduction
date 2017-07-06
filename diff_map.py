import numpy as np
import numpy.linalg as LA
from scipy.spatial import distance
from scipy.sparse.linalg import eigs


class DiffusionMap:
    def __init__(self, eps, step):
        self.eps = eps
        self.step = step

    @staticmethod
    def get_distance_matrix(data):
        return distance.cdist(data, data, 'euclidean')

    def get_kernel(self, data, eps):
        # TODO: use scikit to complete the distances(this may be faster)
        D = distance.cdist(data,data,'euclidean')
        return np.exp(-D ** 2 / eps)

    def get_partial_kernel(self, data, delta, param):
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
            H = self.get_partial_kernel(data, eps, param)
        else:
            H = self.get_kernel(data, eps)
        #H is the kernel matrix
        # TODO : normalization is missing(Laplace Beltrami)
        # Normalization:

        #v = np.sum(H, axis=0)
        #w = np.sum(H, axis=1)
        #mult = 1./np.outer(w, v.T)
        #M = H/v[:,np.newaxis]
        #sqrt_v = np.sqrt(v)
        #den = np.outer(sqrt_v,sqrt_v)
        #M = G/den
        '''
        v = 1./np.sqrt(np.sum(M, axis=0))
        w = 1./np.sqrt(np.sum(M, axis=1))
        mult = np.outer(w, v.T)
        M = np.dot(M, mult)
        '''
        #M is a markov chain
        M = (H.T / H.sum(axis=1)).T
        return M

    def dim_reduction(self, data, ndim, param = None):
        # INPUT:    (N,d) data array: N is number of samples, d dimension
        #           ndim:   desired dimension (ndim<d)
        #           param: contains info about nbhd type for adjancency matrix
        #               knn or eps-nbhd
        # OUTPUT:   (N,ndim) array with reduced dimension

        H = self.get_diffusion_map(data, self.eps, param)
        # We get the first ndim+1 eigenvalues and discard the first one since it is 1
        w, x = eigs(H, k=ndim+1 , which='LM')
        w = w[1:ndim+1]
        print x[:, 0] / x[:,2]
        # Compute data reduction according to Coifman and Lafon :
        # http://www.sciencedirect.com/science/article/pii/S1063520306000546
        # Note: take real part in case of any complex eigenvector/value
        x = (w.real ** self.step) * x[:, 1:ndim+1].real
        #print x[:,0]
        #x = (x.T/x[:,0]).T
        return w, x
