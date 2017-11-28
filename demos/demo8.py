import scipy.spatial.distance as dist
import scipy.sparse.linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gen_data import get_data
from numpy import *
from diff_map import DiffusionMap

## First working draft of parsimonious representation of DM eigenvectors
## Reference:
##  http://ronen.net.technion.ac.il/publications/journal-publications/
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
        A = scipy.linalg.lstsq(Xx2 @ Xx , Xx2)[0]
        L[i,:] = A[0,:]

    # Functional approximation
    FX = L @ Y

    # leave-one-out cross-validation errors
    RES = np.sqrt(np.mean((Y - FX)*(Y - FX))) / np.std(Y)
    return FX, RES


sample = 'two_planes'
data, color = get_data(sample, 1000)

ndim = 12
step = 1
eps=0

# params for eps nearest neighbourhood(epsilon param depends highly on the dataset)
nbhd_param = None
# nbhd_param = {'eps': 1.75}

diff_map = DiffusionMap(step, eps)
diff_map.set_params(data, nbhd_param)
# D contains eigenvalues and V resp. eigenvectors
D, V = diff_map.dm_basis(ndim)


# Linear regression kernel scale
eps_med_scale = 3

# Compute cross-validation error and residuals according to reference
n = V.shape[1]
RES = np.zeros((n-1,1))
RES[0] = 1
for i in range(1,n-1):
    _, RES[i] = local_linear_regression(V[:,i], V[:, :i], eps_med_scale)


RES = np.squeeze(RES)
print("Unsorted Residuals: ",RES)
indices = np.argsort(RES)
indices = indices[::-1]
print("Sorted Residuals: ",RES[indices])

# Reorder eigvalues and eigvectors according to residuals
DD = D[indices]
VV = V[:, indices]
# Take only first two eigvalues/vectors to plot lower dimensional manifold
z = DD[:2]*VV[:,:2]
y = D[:2]*V[:,:2]

fig = plt.figure(figsize=(12, 10))
fig.suptitle("DM vs Parsimonious DM")
ax = fig.add_subplot(231, projection='3d')
ax.set_title("Original 3D data")
ax.scatter(data[:,0], data[:,1], data[:,2], c=color, cmap = plt.cm.coolwarm)

ax = fig.add_subplot(232, projection='3d')
ax.set_title("1st pars. eigendirection in 3D data")
ax.scatter(data[:,0], data[:,1], data[:,2], c=VV[:,0], cmap = plt.cm.Spectral)

ax = fig.add_subplot(233, projection='3d')
ax.set_title("2nd pars. eigendirection in 3D data")
ax.scatter(data[:,0], data[:,1], data[:,2], c=VV[:,1], cmap = plt.cm.Spectral)


ax = fig.add_subplot(234)
ax.set_title("Diffusion Maps")
plt.scatter(y[:, 0], y[:, 1], c=color, cmap=plt.cm.coolwarm)

ax = fig.add_subplot(235)
ax.set_title("Pars. Diffusion Maps")
plt.scatter(z[:, 0], z[:, 1], c=color, cmap=plt.cm.coolwarm)

ax = fig.add_subplot(236)
ax.set_title("Residuals per eigendirection")
xx = range(len(RES))
plt.bar(xx, RES, width=1/1.5, color="blue")
plt.xticks(xx)
plt.show()