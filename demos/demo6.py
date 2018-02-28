import matplotlib.pyplot as plt
import logging
from mpl_toolkits.mplot3d import Axes3D
from diff_map import DiffusionMap
from gen_data import get_data
from pars_rep_dm import compute_res

import numpy as np
# Demo to check parsimonious representation of eigenvectors/values

sample = 'c_curve'#'punc_sphere'
A, color = get_data(sample, 1000)

# target dimension
# For plotting reasons, ndim should be bigger than 5
ndim = 6

# Diff. Map computation and plot
diff_map = DiffusionMap()
diff_map.set_params(A)
w, x = diff_map.dim_reduction(ndim=ndim)

# Plot original 3D data
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(231, projection='3d')
ax.set_title("Original 3D data")
ax.scatter(A[:,0], A[:,1], A[:,2], c=color, cmap = plt.cm.Spectral)

ax = fig.add_subplot(232)
ax.set_title("Diffusion Maps")
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
min_x = min(x[:,0]) - 0.2*np.abs(min(x[:,0]))
min_y = min(x[:,1]) - 0.2*np.abs(min(x[:,1]))
max_x = max(x[:,0]) + 0.2*np.abs(max(x[:,0]))
max_y = max(x[:,1]) + 0.2*np.abs(max(x[:,1]))
plt.axis([min_x, max_x, min_y, max_y])


eps_scale = 5
res,_ = compute_res(x[:,:ndim], eps_scale)
indices = np.argsort(res)
indices = indices[::-1]
y = diff_map.param_from_indices(indices, ndim=ndim)

ax = fig.add_subplot(233)
ax.set_title("Order of eigvec")
plt.scatter(y[:, 0], y[:, 1], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(234)
ax.set_title("C1 vs C2")
indices = [0,1]
x = diff_map.param_from_indices(indices, ndim=ndim)
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(235)
ax.set_title("C1 vs C3")
indices = [0,2]
y = diff_map.param_from_indices(indices, ndim=ndim)
plt.scatter(y[:, 0], y[:, 1], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(236)
ax.set_title("C2 vs C5")
indices = [1,4]
z = diff_map.param_from_indices(indices, ndim=ndim)
plt.scatter(z[:, 0], z[:, 1], c=color, cmap=plt.cm.Spectral)

plt.show()
