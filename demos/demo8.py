import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gen_data import get_data
import numpy as np
from diff_map import DiffusionMap
from pars_rep_dm import compute_res

## First working draft of parsimonious representation of DM eigenvectors
## Reference:
##  https://arxiv.org/abs/1505.06118
## and main matlab code :
##  http://ronen.net.technion.ac.il/files/2016/07/DsilvaACHA.zip

sample = 'two_planes'
data, color = get_data(sample, 1000)
color = np.squeeze(color)

ndim = 10
step = 1
eps = 0

# DM constructor and main parameters
diff_map = DiffusionMap(step, eps)
diff_map.set_params(data)

# D contains DM eigenvalues and V resp. eigenvectors
D, V = diff_map.dm_basis(ndim)

# Linear regression kernel scale
eps_med_scale = 5
# Compute residuals given the eigvectors
RES, _ = compute_res(V, eps_med_scale)

#print("Unsorted Residuals: ",RES)
indices = np.argsort(RES)
indices = indices[::-1]
#print("Sorted Residuals: ",RES[indices])

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