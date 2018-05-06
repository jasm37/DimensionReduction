# demo to test parsimonious eigvectors on a 2d rectangle (from uniform random samples)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diff_map import DiffusionMap
import numpy as np
# Demo to check parsimonious representation of eigenvectors/values

n_samples = 1000
A = np.random.rand(n_samples,2)
sc_x = 4
sc_y = 1
A[:,0] = sc_x * A[:,0]
A[:,1] = sc_y * A[:,1]
color = A[:,0]

# target dimension
ndim = 10
# parameters for diff. maps
step = 1

# Diff. Map computation and plot
# Plot original 3D data
fig = plt.figure(figsize=(8, 20))
ax = fig.add_subplot(131)
ax.set_title("Original 3D data")
ax.scatter(A[:,0], A[:,1], c=color, cmap=plt.cm.Spectral)
ax.set_ylim([-1,2])
ax.set_xlim([-1,5])

ax = fig.add_subplot(132)
ax.set_title("Pars. Diffusion Maps")
dm1 = DiffusionMap(step)
dm1.set_params(A)
w, x = dm1.dim_reduction(ndim, pars=True)
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(133)
ax.set_title("Diffusion Maps")
eps = dm1.eps
dm2 = DiffusionMap(step,eps)
dm2.set_params(A)
w, y = dm2.dim_reduction(ndim, pars=False)
plt.scatter(y[:, 0], y[:, 1], c=color, cmap=plt.cm.Spectral)

plt.show()