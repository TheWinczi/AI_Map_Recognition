import numpy as np
from sklearn.cluster import KMeans
from plotting import plot_convergence


def kmeans(X: np.ndarray):
    """
    K-Means clustering algorithm used for clusterize input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Parameters
    ----------
        X : ndarray
            Array of points coordinates as a list of tuple/lists in shape
            [(x_0, y_0), (x_1, y_1), ...].

    Returns
    -------
    km
        K-Means fitted clusterizer.
    """
    # _check_kmeans_parameters(X)

    km = KMeans(n_clusters=6,
                init="k-means++",
                max_iter=300,
                tol=10 ** (-4))

    y_km = km.fit(X)
    return km


def _check_kmeans_parameters(X: np.ndarray):
    """
    Check all parameters needed in K-Means clustering algorithm.
    Show result on chart.

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of lists/tuples in shape
        [(x_1, y_1), (x_2, y_2), ...]
    """
    errors = []
    n_min, n_max = 4, 20
    for i in range(n_min, n_max + 1):
        km = KMeans(n_clusters=i,
                    init="k-means++",
                    n_init=i + 10,
                    max_iter=300,
                    tol=10 ** (-4),
                    random_state=0)
        y_km = km.fit_predict(X)
        errors.append(km.inertia_)

    plot_convergence(np.array(range(n_min, n_max + 1)), np.array(errors))