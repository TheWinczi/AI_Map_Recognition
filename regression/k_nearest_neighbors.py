import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def k_nearest_neighbors(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    K-Nearest Neighbors regression algorithm used for regression.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    Parameters
    ----------
    X_train : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_train : ndarray
        Array of labels belongs to input X_train data.

    X_test : ndarray {default: None}
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ]. Could be None.

    y_test : ndarray {default: None}
        Array of labels belongs to X_test data. Could be None.

    Returns
    -------
    knn
        Trained regressor ready to regression.
    """
    weight, n_neighbors = _check_knn_params(X_train, y_train, X_test, y_test)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors,
                              weights=weight)
    knn.fit(X_train, y_train)

    return knn


def _check_knn_params(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray):
    """
    Check all parameters needed in K-Nearest Neighbors regression algorithm.
    Show results on plots.

    Parameters
    ----------
    X_train : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_train : ndarray
        Array of labels belongs to input X_train data.

    X_test : ndarray {default: None}
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ]. Could be None.

    y_test : ndarray {default: None}
        Array of labels belongs to X_test data. Could be None.

    Returns
    -------
    (weight, n_neighbors)
        Tuple of best KNN parameters.
    """
    weights = ['distance', 'uniform']
    n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8]

    errors = []
    for weight in weights:
        for n in n_neighbors:
            knn = KNeighborsRegressor(n_neighbors=n,
                                      weights=weight)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            errors.append(sum((y_pred - y_test) ** 2))

    best_index = int(np.argmin(errors))
    best_weight_index = np.floor(best_index / len(n_neighbors)).astype(np.int32)
    best_n_index = best_index % len(n_neighbors)

    return weights[best_weight_index], n_neighbors[best_n_index]
