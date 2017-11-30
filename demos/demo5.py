# Test for projection of outlier(new) points with diffusion maps
# Given an outlier point, compute its projection into the reduced dimension space

import numpy as np
from gen_data import get_data
from diff_map import DiffusionMap
import matplotlib.pyplot as plt
from interpolation import nystrom_ext, rbf_interpolate, poly_rbf
from mpl_toolkits.mplot3d import Axes3D

# Simple script to test diff. maps dimension reduction
samples = 'plane'
num_samples = 1000
A, color = get_data(samples, num_samples)
#A = np.genfromtxt('swiss_roll.txt')

#Choose two random samples and use them as test set
idx = np.random.randint(A.shape[0], size=2)
B = A[idx,:]
A = np.delete(A, idx,0)
color = np.delete(color, idx)

# target dimension
ndim = 10
# parameters for diff. maps
step = 1
eps = 0
# params for k nearest neighbourhood
# params for eps nearest neighbourhood(epsilon param depends highly on the dataset)
#nbhd_param = {'k': 50}
#nbhd_param = None
# nbhd_param = {'eps': 1.75}

dm = DiffusionMap(step, eps)
dm.set_params(A)
#w, x = diff_map.dim_reduction(ndim)

w, x = dm.dm_basis(ndim, pars=True)
#_, indices = compute_res(x)
#dm.permute_indices(indices)
#w, x = dm.dm_basis(n_components=2)
w = w[:2]
x = x[:,:2]
dm_coord = dm.proj_data
pred = []
nys_pred = []
#pred.append(np.squeeze(diff_map.nystrom_ext(B[0,:])))
#pred.append(np.squeeze(diff_map.nystrom_ext(B[1,:])))
#pred.append(dm.nystrom_ext(B[0,:]))
#pred.append(dm.nystrom_ext(B[1,:]))
nys_pred.append(nystrom_ext(B[0,:], A, dm.eps, w, x))
nys_pred.append(nystrom_ext(B[1,:], A, dm.eps, w, x))
coord, _ = poly_rbf(B[0,:], A, dm_coord[:,:2], nnbhd=10)
pred.append(coord)
coord, _ = poly_rbf(B[1,:], A, dm_coord[:,:2], nnbhd=10)
pred.append(coord)

#w, x = dm.dim_reduction(2, pars=True)


interp_pred = []
#print(pred[0])
val, k1 = poly_rbf(pred[0][:2], dm_coord[:,:2], A, nnbhd=50)
#print(val)
interp_pred.append(val)
val, k2 = poly_rbf(pred[1][:2], dm_coord[:,:2], A, nnbhd=50)
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
plt.scatter(dm.proj_data[:,0], dm.proj_data[:,1], c=color, cmap=plt.cm.Spectral)
plt.scatter(pred[0][0], pred[0][1], marker='x', s=40)
plt.scatter(pred[1][0], pred[1][1], marker='v', s=40)
plt.scatter(nys_pred[0][0], nys_pred[0][1], marker='o', s=40)
plt.scatter(nys_pred[1][0], nys_pred[1][1], marker='o', s=40)
plt.scatter(dm_coord[k1, 0], dm_coord[k1, 1], color='b', marker='>', s=40)
plt.scatter(dm_coord[k2, 0], dm_coord[k2, 1], color='y', marker='<', s=40)
plt.grid()
plt.xlabel('phi_1')
plt.ylabel('phi_2')
plt.title('2D representation with Diff. Maps')
fig.tight_layout()
plt.show()