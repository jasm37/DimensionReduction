# Test for projection of outlier(new) points with diffusion maps
# Given an outlier point, compute its projection into the reduced dimension space

import numpy as np
from gen_data import get_data
from diff_map import DiffusionMap
import matplotlib.pyplot as plt
from interpolation import *
from mpl_toolkits.mplot3d import Axes3D

samples = 'gaussian_test'
num_samples = 10000
test, color_2 = get_data(samples, num_samples)

# Simple script to test diff. maps dimension reduction
samples = 'gaussian'
num_samples = 1000
A, color = get_data(samples, num_samples)

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

w, x = dm.dm_basis(ndim, pars=True)
w = w[:2]
x = x[:,:2]
dm_coord = dm.proj_data
pred = []
nys_pred = []

idx = np.random.randint(dm_coord.shape[0], size=2)
C = dm_coord[idx,:]

x = np.linspace(-0.02, 0.02, 30)
y = np.linspace(-0.02, 0.02, 30)

X, Y = np.meshgrid(x, y)
X = X.reshape(-1)
Y = Y.reshape(-1)
data = np.stack((X, Y), axis=1)
list_data = np.ndarray.tolist(data)

eps = 1#0.001
neig = 200
gm_error = 0.001
delta = 0.001

#gh = GeometricHarmonics(dm_coord[:,:2], A, eps, neig, delta=delta)
gh = GeometricHarmonics(A, dm_coord[:,:2], eps, neig, delta=delta)
gh.multiscale_fit(gm_error)

print("Frob. error is ", gh.fro_error)
print("Selected eps is ", gh.eps)
print("Num of eigvectors is ", len(gh.eigval))

mult_val_1 = gh.mult_interpolate(test)
#mult_val_2 = inv_weight(test, data=A, fdata=dm_coord[:,:2])
eps = dm.eps
#mult_val_2 = mult_nystrom_ext(test, data=A, eps=eps, eigvec=dm.proj_data[:,:2], eigval=dm.eigval[:2]).T
mult_val_2 = mult_nystrom_extension(test, data=A, dm=dm).T

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)#, projection='3d')
ax.scatter(dm.proj_data[:,0], dm.proj_data[:,1], c=color, cmap=plt.cm.Spectral)
#ax.scatter(dm.eigvec[:,0], dm.eigvec[:,1], c=color, cmap=plt.cm.Spectral)
plt.title("Original Data")
ax = fig.add_subplot(122)#, projection='3d')
ax.scatter(gh.proj_fdata.T[:,0], gh.proj_fdata.T[:,1], c=color, cmap=plt.cm.Spectral)
plt.title("Geom. Harmonics data")
plt.show()

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(131)#, projection='3d')
ax.scatter(mult_val_1[:,0], mult_val_1[:,1], c=color_2, cmap=plt.cm.Spectral)
plt.title('Geometric Harmonics')

ax = fig.add_subplot(132)#, projection='3d')
ax.scatter(mult_val_2[:,0], mult_val_2[:,1], c=color_2, cmap=plt.cm.Spectral)
plt.title('2nd interpolation method')

ax = fig.add_subplot(133)
plt.scatter(dm_coord[:,0], dm_coord[:,1], c=color, cmap=plt.cm.Spectral)
plt.scatter(C[0,0], C[0,1], c="black",marker='x', s=40)
plt.scatter(C[1,0], C[1,1], c="black",marker='x', s=40)
plt.title('DM data')
plt.grid()
plt.xlabel('phi_1')
plt.ylabel('phi_2')
plt.title('2D representation with Diff. Maps')
fig.tight_layout()
plt.show()