import numpy as np
from sklearn.tree import DecisionTreeRegressor


def decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Decision Tree regression algorithm used for regression.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

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
    tree
        Trained regressor ready to regression.
    """
    criterion, max_depth = _check_tree_params(X_train, y_train, X_test, y_test)

    tree = DecisionTreeRegressor(max_depth=max_depth,
                                 criterion=criterion,
                                 random_state=1)
    tree.fit(X_train, y_train)
    return tree


def _check_tree_params(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray):
    """
    Check all parameters needed in Decision Tree regression algorithm.
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
    (criterion, max_depth)
        Tuple of best Decision Tree parameters.
    """
    criterions = ['mse', 'friedman_mse', 'mae', 'poisson']
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    errors = []
    for depth in max_depths:
        tree = DecisionTreeRegressor(max_depth=depth,
                                     random_state=1)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        errors.append(sum((y_pred - y_test) ** 2))

    best_index = int(np.argmin(errors))
    best_criterion_index = np.floor(best_index / len(max_depths)).astype(np.int32)
    best_depth_index = best_index % len(max_depths)

    return criterions[best_criterion_index], max_depths[best_depth_index]
