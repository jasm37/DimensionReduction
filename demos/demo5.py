# Test for projection of outlier(new) points with diffusion maps
# Given an outlier point, compute its projection into the reduced dimension space

import numpy as np
from gen_data import get_data
from diff_map import DiffusionMap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simple script to test diff. maps dimension reduction
samples = 'two_planes'
num_samples = 1500
A, color = get_data(samples, num_samples)

#Choose two random samples and use them as test set
idx = np.random.randint(A.shape[0], size=2)
B = A[idx,:]
A = np.delete(A, idx,0)
color = np.delete(color, idx)

# target dimension
ndim = 2
# parameters for diff. maps
step = 2
eps = 0
# params for k nearest neighbourhood
# params for eps nearest neighbourhood(epsilon param depends highly on the dataset)
#nbhd_param = {'k': 50}
nbhd_param = None
# nbhd_param = {'eps': 1.75}

diff_map = DiffusionMap(step, eps)
diff_map.set_params(A, nbhd_param)
w, x = diff_map.dim_reduction(ndim)

pred = []
pred.append(diff_map.nystrom_ext(B[0,:]))
pred.append(diff_map.nystrom_ext(B[1,:]))

interp_pred = []
val, k1 = diff_map.rbf_interpolate(pred[0])
#print(val)
interp_pred.append(val)
val, k2 = diff_map.rbf_interpolate(pred[1])
#print(val)
interp_pred.append(val)
#print(interp_pred)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(A[:,0], A[:,1], A[:,2], c = color, cmap = plt.cm.Spectral)
ax.scatter(B[:,0], B[:,1], B[:,2], marker='x', s=40)
ax.scatter(interp_pred[0][0], interp_pred[0][1], interp_pred[0][2], marker='>', s=60)
ax.scatter(interp_pred[1][0], interp_pred[1][1], interp_pred[1][2], marker='<', s=60)
plt.title('Original data')

ax = fig.add_subplot(122)
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)
plt.scatter(pred[0][0], pred[0][1], marker='x', s=40)
plt.scatter(pred[1][0], pred[1][1], marker='x', s=40)
plt.scatter(x[k1, 0], x[k1, 1], color='b', marker='>', s=40)
plt.scatter(x[k2, 0], x[k2, 1], color='y', marker='<', s=40)
plt.xlabel('phi_1')
plt.ylabel('phi_2')
plt.title('2D representation with Diff. Maps')
plt.show()