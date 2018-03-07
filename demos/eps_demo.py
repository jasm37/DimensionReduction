import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gen_data import get_data
import numpy as np
from diff_map import DiffusionMap
from pars_rep_dm import compute_res

# Plot range of "good" epsilons as in
#   "Detecting the slow manifold by anisotropic diffusion maps"-
#   Amit Singer, Radek Erban, I. Kevrekidis, R.R. Coifman

sample = 'torus_curve'
data, color = get_data(sample, 2000)

ndim = 5
step = 1
eps = 0.2

# DM constructor and main parameters
diff_map = DiffusionMap(step, eps)
diff_map.set_params(data)

m_dist = diff_map.sq_distance
#coeffs = np.random.uniform(-14, 14, size=100)

# \epsilon in interval [2^-14, 2^14]
# L(\epsilon) = \sum_i \sum_j W_{ij}(\epsilon)
coeffs = np.arange(-14, 14, 28.0/100.0)
powers = np.power(2, 2*coeffs)

pow_sum = []
for power in powers:
    pow_sum.append(np.sum(np.exp(-m_dist/power)))


plt.plot(powers, pow_sum)
plt.ylabel(r'$L(\epsilon)$')
plt.xlabel(r'$\epsilon$')
plt.loglog()
plt.show()

# D contains DM eigenvalues and V resp. eigenvectors
D, V = diff_map.dm_basis(ndim)

# Linear regression kernel scale
eps_med_scale = 5
# Compute residuals given the eigvectors
RES, _ = compute_res(V, eps_med_scale)

indices = np.argsort(RES)
indices = indices[::-1]

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
ax.scatter(data[:,0], data[:,1], data[:,2], c=color, cmap=plt.cm.coolwarm)

ax = fig.add_subplot(232, projection='3d')
ax.set_title("1st pars. eigendirection in 3D data")
ax.scatter(data[:,0], data[:,1], data[:,2], c=VV[:,0], cmap=plt.cm.Spectral)

ax = fig.add_subplot(233, projection='3d')
ax.set_title("2nd pars. eigendirection in 3D data")
ax.scatter(data[:,0], data[:,1], data[:,2], c=VV[:,1], cmap=plt.cm.Spectral)


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