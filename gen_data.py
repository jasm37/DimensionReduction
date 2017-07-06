import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets


def get_data(name, n_samples):
    # note: gaussian always takes 1600 samples
    noise = 0.1
    mean = np.array([0.0, 0.0])
    covariance = np.array([[2, 0.0], [0.0, 2]])
    return {
        'swiss':datasets.samples_generator.make_swiss_roll(n_samples=n_samples, noise=noise),
        'blobs':datasets.samples_generator.make_blobs(n_samples=n_samples, centers=3, n_features=3, random_state=0),
        's_curve':datasets.samples_generator.make_s_curve(n_samples=n_samples, noise=noise),
        'gaussian':get_3d_clusters(mean, covariance),
    }[name]


def get_3d_clusters(mean, covariance):
    multivariate_normal()
    x = np.linspace(-4.0, 2.5, 40)
    y = np.linspace(-2.5, 4.0, 40)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mean, covariance)
    Z = rv.pdf(pos)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    data = np.stack((X, Y, Z), axis=1)
    return data, Z


def get_clustered_data():
    raise ValueError("2D plot implementation missing")
    ## Gets data for 2D gaussian clusters
    V = []
    # Set mean and variance for clusters
    mu = np.array([[-1.5, 0.0], [1.0, -1.0], [1.0, 1.0]])
    ## Before appending check if matrices are positive semidefinite
    V.append(np.array([[0.2, 0.0], [0.0, 0.2]]))
    V.append(np.array([[0.25, 0.03], [0.03, 0.25]]))
    V.append(np.array([[0.17, -0.01], [-0.01, 0.14]]))
    # no of samples to generate
    no_samples = 300
    nclusters = mu.shape[0]
    # Sample given mean mu and variance V
    X = []
    Y = []
    for i in xrange(nclusters):
        X_temp, Y_temp = np.random.multivariate_normal(mu[i, :], V[i], no_samples).T
        X.append(X_temp)
        Y.append(Y_temp)

    X = np.array(X).reshape(-1)
    Y = np.array(Y).reshape(-1)
    data = np.stack((X, Y), axis=1)
    #h = plt.figure(1)
    '''
    plt.figure(1)
    for i in xrange(nclusters):
       plt.plot(X[i * no_samples:(i + 1) * no_samples], Y[i * no_samples:(i + 1) * no_samples], '.')
    return data
    '''