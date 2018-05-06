import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd

# Demo to show difference between doing PCA with uncentered
# data and centered data: eigenvectors of centered data give
# a representation, eigvec of uncentered are not useful for
# a good representation

disp_x = -1.7
disp_y = 7
x_data = np.asarray([0 , 3.5, 1.3, 5]) + disp_x
y_data = np.asarray([5, 1.5, 3.7, 0]) + disp_y


x = np.asarray(x_data)
y = np.asarray(y_data)
set = np.vstack((x,y))
evec, s, Vt = svd(set)
t = np.linspace(-10,10,20)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
ax.set_title("Original uncentered data")
plt.scatter(x_data, y_data)
plt.plot(evec[0,0]*t, evec[1,0]*t, 'r', label="$PCA_1$" )
plt.plot(evec[0,1]*t, evec[1,1]*t, 'g', label="$PCA_2$")
plt.legend(loc='upper right')
plt.grid()

set = (set.T - np.mean(set,axis=1)).T
evec, s, Vt = svd(set)
ax = fig.add_subplot(122)
ax.set_title("Centered data")
plt.scatter(set[0,:], set[1,:])
plt.plot(evec[0,0]*t, evec[1,0]*t, 'r', label="$PCA_1$")
plt.plot(evec[0,1]*t, evec[1,1]*t, 'g', label="$PCA_2$")
plt.legend(loc='upper right')
plt.grid()
plt.show()