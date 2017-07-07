import numpy as np
from scipy.stats import multivariate_normal
from sklearn import datasets


def get_data(name, n_samples):
    # noise parameter might depened on the dataset to be produced
    noise = 0.1
    mean = np.array([0.0, 0.0])
    covariance = np.array([[2, 0.0], [0.0, 2]])
    return {
        'swiss':datasets.samples_generator.make_swiss_roll(n_samples=n_samples, noise=noise),
        'blobs':datasets.samples_generator.make_blobs(n_samples=n_samples, centers=3, n_features=3, random_state=0),
        's_curve':datasets.samples_generator.make_s_curve(n_samples=n_samples, noise=noise),
        'gaussian':get_3d_clusters(n_samples=n_samples, mean=mean, cov=covariance),
        'plane':get_plane(n_samples=n_samples),
        'two_planes':get_linear_surface(n_samples=n_samples),
        'torus_curve':get_toroidal_helix(n_samples=n_samples),
        'punc_sphere':get_punctured_sphere(n_samples=n_samples)
    }[name]


def get_plane(n_samples):
    n = int(np.sqrt(n_samples))
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-2.0, 2.0, n)
    X, Y = np.meshgrid(x, y)
    Z = 2*X + Y
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    data = np.stack((X, Y, Z), axis=1)
    return data, Z


def get_3d_clusters(n_samples, mean, cov):
    n = int(np.sqrt(n_samples))
    multivariate_normal()
    x = np.linspace(-4.0, 2.5, n)
    y = np.linspace(-2.5, 4.0, n)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    data = np.stack((X, Y, Z), axis=1)
    return data, Z


def get_linear_surface(n_samples):
    n = int(np.sqrt(n_samples/2))
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-2.0, 2.0, n)
    X, Y = np.meshgrid(x, y)
    z_1 = 2*X
    z_2 = 4-2*X
    z_1.reshape(-1)
    z_2.reshape(-1)
    Z = np.stack((z_1,z_2), axis=0)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    X = np.hstack((X,X))
    Y = np.hstack((Y,Y))
    Z = Z.reshape(-1)
    data = np.stack((X, Y, Z), axis=1)
    return data, Z


def get_toroidal_helix(n_samples):
    # Curve on the torus
    # a: radius of the torus
    # b: thickness of the torus
    # k: relation between two angles that parametrize the torus
    noise_mean = 0
    stand_dev = 0.06
    size = n_samples
    a = 3
    b = 1
    k = 5
    t = np.linspace(0,2*np.pi,n_samples)
    X = (a + b*np.cos(k*t))*np.cos(t) + np.random.normal(noise_mean, stand_dev, size)
    Y = (a + b*np.cos(k*t))*np.sin(t) + np.random.normal(noise_mean, stand_dev, size)
    Z = b*np.sin(k*t) + np.random.normal(noise_mean, stand_dev, size)
    data = np.stack((X,Y,Z),axis=1)
    return data, Z


def get_punctured_sphere(n_samples):
    # Uses trigonometric parametrization of the sphere
    rho = 2
    fr = 0.80 # percentage of the sphere to be shown
    n = int(np.sqrt(n_samples))
    p = np.linspace(0, 2*np.pi, n)
    q = np.linspace(0, np.pi*fr, n)
    X = rho*np.outer(np.cos(p),np.sin(q))
    Y = rho*np.outer(np.sin(p),np.sin(q))
    Z = rho*np.outer(np.ones(p.shape[0]),np.cos(q))
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