import numpy as np
from sklearn.cluster import DBSCAN
from plotting import plot_points


def dbscan(X: np.ndarray):
    """
    DBSCAN clustering algorithm used for clusterize input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Parameters
    ----------
        X : ndarray
            Array of points coordinates as a list of tuple/lists in shape
            [(x_0, y_0), (x_1, y_1), ...].

    Returns
    -------
    db
        DBSCAN fitted clusterizer.
    """
    # _check_dbscan_params(X)

    db = DBSCAN(eps=0.175, min_samples=6, metric='manhattan')
    y_db = db.fit_predict(X)
    return db


def _check_dbscan_params(X: np.ndarray):
    """
    Check all parameters needed in DBSCAN clustering algorithm.
    Show result on chart.

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of lists/tuples in shape
        [(x_1, y_1), (x_2, y_2), ...].
    """
    distances = ["euclidean", "l1", "l2", "manhattan", "cosine"]

    for dist in distances:
        for eps in np.linspace(0.1, 0.2, 5):
            db = DBSCAN(eps=eps, metric=dist, min_samples=8)
            y_db = db.fit_predict(X)
            plot_points(X, y_db, title=f"eps={eps}, dist={dist}")
