import numpy as np
import matplotlib.pyplot as plt
from pyprind import ProgBar
from .points import plot_points
from .regions import plot_decision_regions
from .lines import plot_regression


def plot_clusterizers_comparison(clusterizers: list,
                                 X: np.ndarray,
                                 xlabels: list[str] = None,
                                 ylabels: list[str] = None,
                                 titles: list[str] = None,
                                 suptitle: str = None):
    """
    Plot comparison of fitted clusterizers on one figure.

    Parameters
    ----------
    clusterizers : list
        List of fitted clusterizers.

    X : ndarray
        List of tuples/lists of coordinates in shape [(x_0, y_0), ...]

    xlabels : list[str] {default: None}
        List of x axis labels of each clusterizer chart. Could be None.

    ylabels : list[str] {default: None}
        List of y axis labels of each clusterizer chart. Could be None.

    titles : list[str] {default: None}
        List of titles of each clusterizer chart. Could be None.

    suptitle : str {default: None}
        Main title of plot. Could be None.
    """
    plt.figure(figsize=(16, 9))
    for i, cls in enumerate(clusterizers):
        title = None if titles is None else titles[i]
        xlabel = None if xlabels is None else xlabels[i]
        ylabel = None if ylabels is None else ylabels[i]

        plt.subplot(3, 1, i + 1)
        plot_points(X, cls.labels_,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    new_fig=False, show=False)

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_classifiers_comparison(classifiers: list,
                                X: np.ndarray,
                                y: np.ndarray,
                                xlabels: list[str] = None,
                                ylabels: list[str] = None,
                                titles: list[str] = None,
                                suptitle: str = None):
    """
    Plot comparison of trained classifiers on one figure.

    Paramters
    ---------
    classifiers : list
        List of trained classifiers.

    X : ndarray
        List of tuples/lists of input data in shape [(x_0, x_1), (x_0, x_1) ...]

    y : ndarray
        Labels belong to input X elements.

    xlabels : list[str] {default: None}
        List of x axis labels of each classifier chart. Could be None.

    ylabels : list[str] {default: None}
        List of y axis labels of each classifier chart. Could be None.

    titles : list[str] {default: None}
        List of titles of each classifier chart. Could be None.

    suptitle : str {default: None}
        Main title of plot. Could be None.
    """
    print('\033[91m' + "Plotting classifiers comparison..." + '\033[0m')
    pb = ProgBar(len(classifiers)+1)
    pb.update()

    plot_cols = np.floor(np.sqrt(len(classifiers))).astype(np.int32)
    plot_rows = np.ceil(len(classifiers)/plot_cols).astype(np.int32)

    plt.figure(figsize=(16, 9))
    for i, classifier in enumerate(classifiers):
        title = None if titles is None else titles[i]
        xlabel = None if xlabels is None else xlabels[i]
        ylabel = None if ylabels is None else ylabels[i]

        plt.subplot(plot_rows, plot_cols, i+1)
        plot_decision_regions(classifier,
                              X, y,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              new_fig=False, show=False)

        pb.update()

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def plot_regressors_comparsion(regressors: list,
                               X: np.ndarray, y_true: np.ndarray,
                               test_func=None,
                               titles: list[str] = None,
                               xlabels: list[str] = None,
                               ylabels: list[str] = None,
                               suptitle: str = None):
    """
    Plot comparison of trained regressors on one figure.

    Parameters
    ---------
    regressors : list
       List of trained regressors.

    X : ndarray
       List of tuples/lists of input data in shape [(x_0, x_1), (x_0, x_1) ...]

    y_true : ndarray
       True values according to X tuples/lists.

    test_func
        Function used as a input_fn in Tensorflow estimators.
        Is ignored in the rest of regressors objects.

    xlabels : list[str] {default: None}
        List of x axis labels of each regressor chart. Could be None.

    ylabels : list[str] {default: None}
       List of y axis labels of each regressor chart. Could be None.

    titles : list[str] {default: None}
       List of titles of each regressor chart. Could be None.

    suptitle : str {default: None}
       Main title of plot. Could be None.
    """

    plot_cols = np.floor(np.sqrt(len(regressors))).astype(np.int32)
    plot_rows = np.ceil(len(regressors) / plot_cols).astype(np.int32)

    plt.figure(figsize=(16, 9))
    for i, regressor in enumerate(regressors):
        title = None if titles is None else titles[i]
        xlabel = None if xlabels is None else xlabels[i]
        ylabel = None if ylabels is None else ylabels[i]

        plt.subplot(plot_rows, plot_cols, i + 1)
        plot_regression(regressor, X, y_true,
                        func=test_func,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        new_fig=False, show=False,
                        kind="reduce")

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
