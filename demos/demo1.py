# For 3D plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diff_map import DiffusionMap
from gen_data import get_data

if __name__ == '__main__':

    A, color = get_data('gaussian', 1500)
    print A.shape
    print color.shape

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(A[:,0], A[:,1], A[:,2], c = color, cmap = plt.cm.Spectral)

    # target dimension
    ndim = 2
    # parameters for diff. maps
    step = 2
    eps = 10
    # params for k nearest neighbourhood
    # params for eps nearest neighbourhood(epsilon param depends highly on the dataset)
    param = {'k': 50}
    # param = {'eps': 1.75}

    diff_map = DiffusionMap(eps, step)
    w, x = diff_map.dim_reduction(A, ndim, param=param)
    #w, x = dim_reduction(A, ndim, eps, step)

    ax = fig.add_subplot(122)
    plt.scatter(x[:, 0], x[:, 1], c=color, cmap=plt.cm.Spectral)

    plt.show()

    '''
    # For markov chain visualization(as in Coifmans Diff. Maps paper)
    H = get_markov_chain(A,0.7)
    H_8 = LA.matrix_power(H, 8)
    a = plt.figure(1)
    plt.title('H')
    plt.pcolor(H)
    plt.colorbar()
    a.show()

    c = plt.figure(3)
    plt.title('H^8')
    plt.pcolor(H_8)
    plt.colorbar()
    c.show()
    '''
