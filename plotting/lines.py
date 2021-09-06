import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.estimator import DNNRegressor


def plot_convergence(X: np.ndarray,
                     y: np.ndarray,
                     title: str = None,
                     xlabel: str = None,
                     ylabel: str = None,
                     color: float = "red"):
    """
    Plot convergence.

    Parameters
    ----------
        X : ndarray
            List of x coordinates.
        y : ndarray
            List of y coordinates.
        xlabel : str {default: None}
            String which will be using as a xlabel on chart
        ylabel : str {default: None}
            String which will be using as an ylabel on chart
        title : str {default: None}
            String which will be using as a title on chart
        color : str {default: red}
            String representing line color.
    """

    fig = plt.figure()
    plt.plot(X, y, '-o', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_learning_history(history: dict[str, list]):
    """
    Plot learning history in one plot.

    Parameters
    ----------
    history : dict
        Dictionary object stores history of learning e.g.
        history = {'accuracy' : [....],
                   'loss': [...]}
    """
    num_epochs = len(history['accuracy'])

    plt.figure(figsize=(16, 9))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), history['loss'])
    plt.title('Loss Function')
    plt.xlabel('epoch')
    plt.ylabel('value')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), history['accuracy'])
    plt.title('Learning Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('value')

    plt.tight_layout()
    plt.show()


def plot_regression(regressor,
                    X: np.ndarray, y_true: np.ndarray,
                    func=None,
                    title: str = None,
                    xlabel: str = None,
                    ylabel: str = None,
                    new_fig: bool = True,
                    show: bool = True,
                    kind: str = "both"):
    """
    Plot regression result on one plot.

    Parameters
    ----------
    regressor
        Trained regressor ready to predict
    X : ndarray
        List of list/tuples of input data to predict
    y_true : ndarray
        List of true values
    func
        Function used as input_fn in Tensorflow estimators.
        In the rest of regressors is ignored.
    title : str {default: None}
        String value of plot title
    xlabel : str {default: None}
        String value of plot xlabel
    ylabel : str {default: None}
        String value of plot ylabel
    new_fig : bool {default: True}
       Is new figure has to be created
    show : bool {default: True}
        Is plot has to be shown
    kind : str {default: "both"}
        Which kid of plot has to be created. Has to be
        element of list ["both", "normal", "reduce"]
    """

    assert kind in ["both", "normal", "reduce"], "Bad kind argument value"

    if new_fig:
        plt.figure(figsize=(16, 9))

    if isinstance(regressor, DNNRegressor):
        df_test = _cast_to_dataframe(X, y_true)
        predictioner = regressor.predict(input_fn=lambda: func(df_test))
        y_pred_normal = np.array([value['predictions'][0] for value in iter(predictioner)])
    else:
        y_pred_normal = regressor.predict(X)

    y_avg, y_std = np.average(y_true), np.std(y_true)
    indices = np.logical_and(y_true <= y_avg + y_std, y_true >= y_avg - y_std)
    X_reduce, y_true_reduce, y_pred_reduce = X[indices, :], y_true[indices], y_pred_normal[indices]

    if kind == "both":
        plt.subplot(1, 2, 1)
        _draw_regression(y_true, y_pred_normal)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.subplot(1, 2, 2)
        _draw_regression(y_true_reduce, y_pred_reduce)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.suptitle(title)
    elif kind == "normal":
        _draw_regression(y_true, y_pred_normal)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    elif kind == "reduce":
        _draw_regression(y_true_reduce, y_pred_reduce)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()


def _draw_regression(y_true: np.ndarray, y_pred: np.ndarray):
    plt.plot(y_pred, '-', c='red', label="Regression Values")
    plt.plot(y_true, c='blue', label="Real Values")

    errors = y_true - y_pred

    regression_summary = f"ERRORS SUMMARY\nsum = {sum(errors**2)}\navg={np.average(errors)}\nstd={np.std(errors)}"
    plt.text(x=len(y_true)/2, y=max(y_true), s=regression_summary, ha='center', va='top')


def _cast_to_dataframe(X: np.ndarray, y: np.ndarray):
    df = pd.DataFrame.from_dict({
        "x": X[:, 0],
        "y": X[:, 1],
        "time": y[:]
    })
    return df
