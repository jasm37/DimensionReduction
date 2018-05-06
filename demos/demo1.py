import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diff_map import DiffusionMap
from gen_data import get_data


# Simple script to test diff. maps dimension reduction
samples = 'curve_3d'
num_samples = 1500
A, color = get_data(samples, num_samples)
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(A[:,0], A[:,1], A[:,2], c = color, cmap = plt.cm.Spectral)
plt.title('Original data')
# target dimension
ndim = 2
# parameters for diff. maps
step = 2
eps = 0 # if eps = 0, then eps is computed as the median

# params for k nearest neighbourhood
# params for eps nearest neighbourhood(epsilon param depends highly on the dataset)
nbhd_param = None
#nbhd_param = {'k': 10}
nbhd_param = {'eps': 0.05}

diff_map = DiffusionMap(step, eps)
diff_map.set_params(A, nbhd_param)
w, x = diff_map.dim_reduction(ndim)

x -= np.mean(x, axis=0)
x /= np.std(x, axis=0)

ax = fig.add_subplot(122)
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
# phi_i are eigvectors
plt.xlabel('$\phi_1$')
plt.ylabel('$\phi_2$')
plt.title('2D representation with Diff. Maps')
plt.show()
