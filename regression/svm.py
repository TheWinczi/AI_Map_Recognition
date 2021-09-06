import numpy as np
from sklearn.svm import SVR


def svm_regression(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Support Vector Regression algorithm used for regression.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

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
    svm
        Trained regressor ready to regression.
    """
    kernel, C = _check_svr_params(X_train, y_train, X_test, y_test)

    svr = SVR(C=C,
              kernel=kernel)
    svr.fit(X_train, y_train)
    return svr


def _check_svr_params(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray):
    """
    Check all parameters needed in Support Vector Regression (SVR) algorithm.

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
    (kernel, C)
        Tuple of best KNN parameters.
    """
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    Cs = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

    errors = []
    for kernel in kernels:
        for C in Cs:
            svr = SVR(C=C,
                      kernel=kernel)
            svr.fit(X_train, y_train)

            y_pred = svr.predict(X_test)
            errors.append(sum((y_pred - y_test) ** 2))

    best_index = int(np.argmin(errors))
    best_kernel_index = np.floor(best_index / len(Cs)).astype(np.int32)
    best_C_index = best_index % len(Cs)

    return kernels[best_kernel_index], Cs[best_C_index]
