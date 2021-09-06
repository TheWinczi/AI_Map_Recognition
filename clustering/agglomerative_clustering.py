import numpy as np
from sklearn.cluster import AgglomerativeClustering
from plotting import plot_points


def agg(X: np.ndarray):
    """
    Agglomarative Clustering algorithm used for clusterize input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

    Parameters
    ----------
        X : ndarray
            Array of points coordinates as a list of tuple/lists in shape
            [(x_0, y_0), (x_1, y_1), ...].

    Returns
    -------
    agg
        AgglomerativeClustering fitted clusterizer.
    """
    # _check_agg_params(X)

    agg = AgglomerativeClustering(n_clusters=6,
                                  affinity='euclidean',
                                  linkage='ward')
    y_agg = agg.fit_predict(X)
    return agg


def _check_agg_params(X: np.ndarray):
    """
    Check all parameters needed in AgglomerativeClustering algorithm.
    Show result on chart.

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of lists/tuples in shape
        [(x_1, y_1), (x_2, y_2), ...].
    """
    n_min, n_max = 6, 6
    linkages = ['ward', 'complete', 'average', 'single']
    affinities = ["euclidean", "l1", "l2", "manhattan", "cosine"]

    for linkage in linkages:
        for aff in affinities:
            for i in range(n_min, n_max + 1):
                agg = AgglomerativeClustering(n_clusters=6,
                                              affinity=aff,
                                              linkage=linkage)
                y_agg = agg.fit_predict(X)
                plot_points(X, y_agg, title=f"link={linkage}, n={i}, aff={aff}")
