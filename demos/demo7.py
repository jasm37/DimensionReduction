# demo to test parsimonious eigvectors on a 2d rectangle (from uniform random samples)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diff_map import DiffusionMap
from gen_data import get_data
from pars_rep_dm import compute_res
import numpy as np
# Demo to check parsimonious representation of eigenvectors/values

sample = 'gaussian'
A = np.random.rand(700,2)
sc_x = 4
sc_y = 1
A[:,0] = sc_x * A[:,0]
A[:,1] = sc_y * A[:,1]


# target dimension
ndim = 10
# parameters for diff. maps
step = 1

# Diff. Map computation and plot
dm1 = DiffusionMap(step)
dm1.set_params(A)
w, x = dm1.dim_reduction(ndim,pars=True)
#diff_map.compute_eigdecomp(ndim)
#diff_map.dim_reduction(ndim,pars=False)
# Plot original 3D data
fig = plt.figure(figsize=(8, 20))
ax = fig.add_subplot(131)
ax.set_title("Original 3D data")
ax.scatter(A[:,0], A[:,1])
ax.set_ylim([-1,2])
ax.set_xlim([0,4])


ax = fig.add_subplot(132)
ax.set_title("Pars. Diffusion Maps")
plt.scatter(x[:, 0], x[:, 1], cmap=plt.cm.Spectral)

ax = fig.add_subplot(133)
ax.set_title("Diffusion Maps")
eps = dm1.eps
dm2 = DiffusionMap(step,eps)
dm2.set_params(A)
w, x = dm2.dim_reduction(ndim,pars=False)
plt.scatter(x[:, 0], x[:, 1], cmap=plt.cm.Spectral)

plt.show()
'''
w, x = diff_map.dm_basis(n_components=ndim)


eps_scale = 1
#eps_scale = diff_map.eps
res,_ = compute_res(x[:ndim].T, eps_scale)
print("res computed,",res)
indices = np.argsort(res)
indices = indices[::-1]
print("res sorted,",res[indices])
#print(indices)
y = diff_map.param_from_indices(indices, ndim=ndim)
ax = fig.add_subplot(233)
ax.set_title("Order of eigvec")
plt.scatter(y[:, 0], y[:, 1], cmap=plt.cm.Spectral)
#plt.plot(res)


#fig2 = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(234)
ax.set_title("C1 vs C2")
indices = [0,1]
x = diff_map.param_from_indices(indices, ndim=ndim)
plt.scatter(x[:, 0], x[:, 1], cmap=plt.cm.Spectral)

ax = fig.add_subplot(235)
ax.set_title("C1 vs C3")
indices = [0,2]
y = diff_map.param_from_indices(indices, ndim=ndim)
plt.scatter(y[:, 0], y[:, 1], cmap=plt.cm.Spectral)

ax = fig.add_subplot(236)
ax.set_title("C1 vs C4")
indices = [0,3]
z = diff_map.param_from_indices(indices, ndim=ndim)
plt.scatter(z[:, 0], z[:, 1], cmap=plt.cm.Spectral)

plt.show()
'''