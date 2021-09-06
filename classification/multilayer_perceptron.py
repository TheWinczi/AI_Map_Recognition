import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from plotting import *


def mlp(X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    MultiLayer Perceptron (MLP) classification algorithm used for classification input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

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
    mlp
        Trained classifier ready to predict.
    """
    # _check_mlp_params(X_train, y_train)

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-2, activation='relu',
                        hidden_layer_sizes=(50,), random_state=1,
                        tol=10**(-3))
    mlp.fit(X_train, y_train)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = mlp.predict(X_test)
        print(f"MultiLayer Perceptron test accuracy: {accuracy_score(y_test, y_pred)}")

    return mlp


def _check_mlp_params(X: np.ndarray, y: np.ndarray):
    """
    Check all parameters needed in Multilayer Perceptron (MLP) classification algorithm.
    Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X data.
    """
    activations = ["tanh", "relu", "identity", "logistic"]

    for activation in activations:
        for i in range(30, 61, 10):
            mlp = MLPClassifier(solver='lbfgs', alpha=1e-2, activation=activation,
                                hidden_layer_sizes=(i,), random_state=1,
                                tol=10**(-3))
            mlp.fit(X, y)
            plot_decision_regions(mlp, X, y,
                                  xlabel="x axis", ylabel="y axis",
                                  title=f"MLP, h.layers=({i},), activation={activation}")
