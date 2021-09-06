import numpy as np
from sklearn.preprocessing import StandardScaler
from plotting import plot_regression, plot_regressors_comparsion
from .neural_network import neural_network, eval_input_fn
from .k_nearest_neighbors import k_nearest_neighbors
from .decision_tree import decision_tree
from .svm import svm_regression


def regression(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Train regressor using input array of data using the best
    regression algorithm.

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
    regressor
        Trained regressor ready to regression.
    """
    svm_ = svm_regression(X_train, y_train, X_test, y_test)
    return svm_


def try_all_regressors(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray = None, y_test: np.ndarray = None,
                       compare: bool = True):
    """
    Try all regression algorithm which were implemented.

    Parameters
    ----------
    X_train : ndarray
        Array of data (points) as a list o tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_train : ndarray
        Array of labels belongs to X_train data.

    X_test : ndarray {default: None}
        Array of data (points) as a list o tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ]. Could be None.

    y_test : ndarray {default: None}
       Array of labels belongs to X_test data. Could be None.

    compare : bool {default: True}
       Are classifiers needed to be compared on some plots.

    Returns
    -------
    regressor
       The best trained regressor ready to regression.
    """
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    dnn_ = neural_network(X_train, y_train, X_test, y_test)
    # plot_regression(dnn_, X_test, y_test,
    #                 title="DNN Regression results")

    svm_ = svm_regression(X_train, y_train, X_test, y_test)
    # plot_regression(svm_, X_test, y_test,
    #                 title="SVM Regression results")

    tree_ = decision_tree(X_train, y_train, X_test, y_test)
    # plot_regression(tree_, X_test, y_test,
    #                 title="Decision Tree Regression results")

    knn_ = k_nearest_neighbors(X_train, y_train, X_test, y_test)
    # plot_regression(knn_, X_test, y_test,
    #                 title="KNN Regression results")

    if compare:
        regressors = [dnn_, svm_, tree_, knn_]
        plot_regressors_comparsion(regressors,
                                   X_test, y_test,
                                   titles=["Deep Neural Network", "SVM", "Decision Tree", "KNN"],
                                   suptitle="Regression algorithms comparison",
                                   test_func=eval_input_fn)

    return svm_
