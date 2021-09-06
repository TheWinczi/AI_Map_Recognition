import numpy as np
from sklearn.cluster import SpectralClustering
from plotting import plot_points


def spectral_clustering(X: np.ndarray):
    """
    SpectralClustering clustering algorithm used for clusterize input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of tuple/lists in shape
        [(x_0, y_0), (x_1, y_1), ...].

    Returns
    -------
    sc
        SpectralClustering fitted clusterizer.
    """
    # _check_spectral_clustering_params(X)

    sc = SpectralClustering(n_clusters=6,
                            gamma=0.1,
                            assign_labels='discretize',
                            random_state=0)
    y_sc = sc.fit_predict(X)
    return sc


def _check_spectral_clustering_params(X: np.ndarray):
    """
    Check all parameters needed in SpectralClustering clustering algorithm.
    Show result on chart.

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of lists/tuples in shape
        [(x_1, y_1), (x_2, y_2), ...].
    """
    for n in range(7, 16):
        sc = SpectralClustering(n_clusters=6,
                                gamma=0.1,
                                assign_labels='discretize',
                                random_state=0)
        y_sc = sc.fit_predict(X)
        plot_points(X, y_sc, title=f"n={n}")