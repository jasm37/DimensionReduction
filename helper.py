import scipy.signal as scs
import scipy.spatial.distance as dist
import numpy as np

def smooth(y):
    """
    Smoothens high frequencies of data
    :param y: "noisy" data
    :return: smoothened data
    """
    y_smooth = scs.medfilt(y)
    # There are some issues at the beginning and end
    y_smooth[0] = y[0]
    y_smooth[-1] = y[-1]
    return y_smooth


def get_local_nbhd(data, x, nnbhd):
    """
    Computes local neighbourhood of a set of points
    :param data: Data from which neighbours are selected
    :param x: Set of points for which neighbourhoods are to be computed. Must satisfy len(x) > 1
    :param nnbhd: Number of neighbours per point to consider
    :return: List of neighbours indices from data to x
    """
    # Compute distance matrix
    dist_vec = dist.cdist(x, data, 'sqeuclidean')

    # Add indices to the set(repeated indices count as one)
    nbhd_set = set()
    for i in range(x.shape[0]):
        temp = np.argsort(dist_vec[i, :])[:nnbhd]
        nbhd_set.update(temp)
    nbhd_list = list(nbhd_set)

    # Optional: Take a random sample from the nbhd list!
    #           This makes the Geometric Harmonics procedure faster
    # 0 < factor < 1
    factor = 0.25
    #nbhd_list = np.random.choice(nbhd_list, int(len(nbhd_list)*factor))
    return nbhd_list