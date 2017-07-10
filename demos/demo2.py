# Diff. Maps vs other methods:
# For 3D plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diff_map import DiffusionMap
from gen_data import get_data
from sklearn.decomposition import PCA

# Compares Diff. Map dim. reduction and PCA

# Pick here which data to generate for the coomparison
sample = 'gaussian'
A, color = get_data(sample, 1000)

# Plot original 3D data
fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(131, projection='3d')
ax.set_title("Original 3D data")
ax.scatter(A[:,0], A[:,1], A[:,2], c = color, cmap = plt.cm.Spectral)

# target dimension
ndim = 2
# parameters for diff. maps
step = 1
eps = 0
# params for k nearest neighbourhood
# params for eps nearest neighbourhood(epsilon param depends highly on the dataset)
#nbhd_param = {'k': 50}
# param = {'eps': 1.75}

# Diff. Map computation and plot
diff_map = DiffusionMap(step, eps)
#diff_map.set_params(A, nbhd_param)
diff_map.set_params(A)
w, x = diff_map.dim_reduction(ndim)

ax = fig.add_subplot(132)
ax.set_title("Diffusion Maps")
plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)

# PCA computation and plot
pca = PCA(n_components=2)
U =pca.fit_transform(A)

ax = fig.add_subplot(133)
ax.set_title("PCA")
plt.scatter(U[:, 0], U[:, 1], c=color, cmap=plt.cm.Spectral)

plt.show()
