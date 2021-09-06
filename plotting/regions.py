import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .utilities import _get_colors, _get_markers


def plot_decision_regions(classifier,
                          X: np.ndarray,
                          y: np.ndarray = None,
                          xlabel: str = None,
                          ylabel: str = None,
                          title: str = None,
                          labels: list[str] = None,
                          draw_regions: bool = True,
                          new_fig: bool = True,
                          show: bool = True):
    """
    Plot decision regions using trained cluster object.

    Parameters
    ----------
    classifier : Classifier
        Classifier which is used for predictions. Could be list of classifiers.

    X : ndarray
        List of coordinates list or tuples (x, y) when x and y
        are coordinates in euclides space.

    y : ndarray {default: None}
        List of labels belongs to input X tuples/lists. Could be None.

    xlabel : str {default: None}
        Plot x axis label. Could be None.

    ylabel : str {default: None}
        Plot y axis label. Could be None.

    title : str {default: None}
        Plot title. Could be None.

    labels: list {default: None}
        List of strings which will be used to create plot legend. Could be None.

    draw_regions: bool {default: True}
        Are decisions regions have to be shown.

    new_fig : bool {default: True}
        Is new figure has to be created.

    show : bool {default: True}
        Is plot has to be shown.
    """

    # create list of unique classes on which X elements can belong
    if y is None or len(y) != len(X):
        y = classifier.predict(X)

    classes = np.unique(y)
    N = len(classes)

    if new_fig:
        plt.figure(figsize=(16, 9))

    # plotting decision regions - regions to which points are classified
    # by a given classifier. Different colors are used for each region
    if draw_regions:
        colors = _get_colors(N, kind="str")
        cmap = ListedColormap(colors)

        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        resolution = 0.01
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

        # if result of prediction is list of probabilities of belongs - reduce
        # these list to indices of max value of belong probability
        if isinstance(Z[0], np.ndarray) or isinstance(Z[0], list) or isinstance(Z[0], tuple):
            Z = np.array(list(map(lambda item: item.index(max(item)), Z.tolist())))

        Z = Z.reshape(xx1.shape)

        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    if labels is None:
        labels = range(N)

    # creating and plotting point using different markers and
    # markers colors for all (X data) labels
    markers = _get_markers(N)
    colors = _get_colors(N)

    for i, clazz in enumerate(classes):
        Xs = X[y == clazz][:, 0]
        Ys = X[y == clazz][:, 1]
        plt.scatter(Xs, Ys, marker=markers[i], cmap=colors[i], label=labels[i])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.legend()

    if show:
        plt.show()
