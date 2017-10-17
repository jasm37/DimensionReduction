# demo to test parsimonious eigvectors on a 2d rectangle (from uniform random samples)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diff_map import DiffusionMap
from gen_data import get_data
from pars_rep_dm import compute_res
import numpy as np
# Demo to check parsimonious representation of eigenvectors/values

sample = 'gaussian'
A = np.random.rand(500,2)
sc_x = 4
sc_y = 1
A[:,0] = sc_x * A[:,0]
A[:,1] = sc_y * A[:,1]




# target dimension
ndim = 2
# parameters for diff. maps
step = 1
# when eps = 0 then the class computes an appropriate eps according to the dataset
#eps = 22

# params for k nearest neighbourhood
# params for eps nearest neighbourhood(epsilon param depends highly on the dataset)
#nbhd_param = {'k': 30}
#nbhd_param = {'eps': 1.75}

# Diff. Map computation and plot
diff_map = DiffusionMap(step)
#diff_map.set_params(A, nbhd_param)
diff_map.set_params(A)
w, x = diff_map.dim_reduction(ndim)
diff_map.compute_eigdecomp(ndim)

# Plot original 3D data
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(231)
ax.set_title("Original 3D data")
ax.scatter(A[:,0], A[:,1])
ax.set_ylim([-1,2])
ax.set_xlim([0,4])

ax = fig.add_subplot(232)
ax.set_title("Diffusion Maps")
plt.scatter(x[:, 0], x[:, 1], cmap=plt.cm.Spectral)
w, x = diff_map.dm_basis(n_components=4)
eps_scale = 0.00001
res = np.squeeze(compute_res(x[:10].T, eps_scale))
print("res computed,",res)
indices = np.argsort(res)
indices = indices[::-1]
print("res sorted,",res[indices])
#print(indices)
y = diff_map.param_from_indices(indices, ndim=2)
ax = fig.add_subplot(233)
ax.set_title("Order of eigvec")
plt.scatter(y[:, 0], y[:, 1], cmap=plt.cm.Spectral)
#plt.plot(res)


#fig2 = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(234)
ax.set_title("C1 vs C2")
indices = [0,1]
x = diff_map.param_from_indices(indices, ndim=2)
plt.scatter(x[:, 0], x[:, 1], cmap=plt.cm.Spectral)

ax = fig.add_subplot(235)
ax.set_title("C1 vs C3")
indices = [0,2]
y = diff_map.param_from_indices(indices, ndim=2)
plt.scatter(y[:, 0], y[:, 1], cmap=plt.cm.Spectral)

ax = fig.add_subplot(236)
ax.set_title("C1 vs C4")
indices = [0,3]
z = diff_map.param_from_indices(indices, ndim=2)
plt.scatter(z[:, 0], z[:, 1], cmap=plt.cm.Spectral)

'''
ax = fig.add_subplot(237)
ax.set_title("C2 vs C4")
indices = [1,3]
w = diff_map.param_from_indices(indices, ndim=2)
plt.scatter(w[:, 0], w[:, 1], c=color, cmap=plt.cm.Spectral)
'''
plt.show()
