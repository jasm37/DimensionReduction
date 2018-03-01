import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diff_map import DiffusionMap
from gen_data import get_data

sample = 'swiss'#'punc_sphere'
A, color = get_data(sample, 2000)

#nbhd_param = {'eps': 3}
#nbhd_param = {'k': 100}
# target dimension
# For plotting reasons, ndim should be bigger than 5
ndim = 12

# Diff. Map computation and plot
diff_map = DiffusionMap()
diff_map.set_params(A)#, nbhd_param=nbhd_param)
w, x = diff_map.dim_reduction(ndim=ndim, pars=True)
print(w)
#x = diff_map.eigvec

# Plot original 3D data
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(231, projection='3d')
ax.set_title("Original 3D data")
ax.scatter(A[:,0], A[:,1], A[:,2], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(232)
ax.set_title("Diffusion Maps")
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)


ax = fig.add_subplot(233)
ax.set_title("Order of eigvec")
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(234)
ax.set_title("C1 vs C3")
plt.scatter(x[:, 0], x[:, 2], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(235)
ax.set_title("C1 vs C4")
plt.scatter(x[:, 1], x[:, 3], c=color, cmap=plt.cm.Spectral)

ax = fig.add_subplot(236)
ax.set_title("C2 vs C5")
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)

plt.show()