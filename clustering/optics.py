import numpy as np
from sklearn.cluster import OPTICS
from plotting import plot_points


def optics(X: np.ndarray):
    """
    OPTICS clustering algorithm used for clusterize input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of tuple/lists in shape
        [(x_0, y_0), (x_1, y_1), ...].

    Returns
    -------
    opt
        OPTICS fitted clusterizer.
    """
    # _check_optics_params(X)

    opt = OPTICS(min_samples=50,
                 min_cluster_size=50,
                 eps=0.25)
    y_opt = opt.fit_predict(X)
    return opt


def _check_optics_params(X: np.ndarray):
    """
    Check all parameters needed in OPTICS clustering algorithm.
    Show result on chart.

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of lists/tuples in shape
        [(x_1, y_1), (x_2, y_2), ...].
    """
    for i in np.linspace(0.2, 1.0, 10):
        opt = OPTICS(min_cluster_size=8,
                     eps=i)

        y_opt = opt.fit_predict(X)
        plot_points(X, y_opt, title=f"eps={i}")