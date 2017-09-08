import numpy as np
from gen_data import get_data
from sklearn.decomposition import PCA
from numpy.linalg import svd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given data, use its POD/PCA decomp. to get a good approximation

sample = "punc_sphere"
A, color = get_data(sample, 1500)

n= A.shape[1]
mean_t = np.sum(A, axis=1) / n
centered_temp = (A.T - mean_t).T

evec, s, Vt = svd(centered_temp)
n_comp = 4
evec = evec[:, 0:n_comp]
print(s[:3])
# gram contains matrix of coefficients :
# x = sum_i coeff_i * phi_i
# where x is a datapoint, phi_i is eigenvector
#gram = evec.T @ centered_temp
gram = np.matmul(evec.T, centered_temp)
print('Eigenvalues reduced shape is', evec.shape)
#approx = evec @ gram
approx = (np.matmul(evec, gram).T + mean_t).T
print(approx.shape)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(131, projection='3d')
ax.scatter(approx[:, 0], approx[:, 1], approx[:, 2],c=color, cmap=plt.cm.Spectral)
plt.title('Approximated data')
ax = fig.add_subplot(132, projection='3d')
ax.scatter(A[:, 0], A[:, 1], A[:,2], c=color, cmap=plt.cm.Spectral)
plt.title('Original data')

err = A- approx
ax = fig.add_subplot(133, projection='3d')
ax.scatter(err[:, 0], err[:, 1], err[:,2], c=color, cmap=plt.cm.Spectral)
plt.title('error')
print("Error is",np.linalg.norm(err, 'fro'))
plt.show()