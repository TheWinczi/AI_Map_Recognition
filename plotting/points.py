import numpy as np
import matplotlib.pyplot as plt
from .utilities import _get_colors, _get_markers


def plot_points(X: np.ndarray,
                y: np.ndarray = None,
                title: str = None,
                xlabel: str = None,
                ylabel: str = None,
                zlabel: str = None,
                new_fig: bool = True,
                show: bool = True):
    """
    Plot points in n-dimension area. Can plot 2D and 3D points.

    Parameters
    ----------
    X : ndarray
        list of coordinates list or tuples (x, y) when x and y
        are coordinates in euclides space.

    y : ndarray {default: None}
        List of labels which points belongs to. Defines how many
        colors and markers will bu used - different label is equal
        to different marker and point color

    xlabel : str {default: None}
        Plot x axis label. Could be None.

    ylabel : str {default: None}
        Plot y axis label. Could be None.

    zlabel : str {default: None}
        Plot z axis label. Could be None.

    title : str {default: None}
        Plot title. Could be None.

    new_fig : bool {default: True}
        Is new figure has to be created.

    show : bool {default: True}
        Is new plot has to be shown after creation.
    """

    if y is not None and len(y) == len(X):
        markers = _get_markers(len(np.unique(y)))
        colors = _get_colors(len(np.unique(y)), kind="str")
    else:
        markers = _get_markers(1)
        colors = _get_colors(1)

    dim = len(X[0])

    if new_fig or dim == 3:
        fig = plt.figure(figsize=(16, 9))

    if dim == 2:
        _plot_2D_points(X, y, colors, markers)
    elif dim == 3:
        ax = _plot_3D_points(X, y, colors, markers, fig)
        ax.set_zlabel(zlabel)
    else:
        return

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if show:
        plt.show()


def _plot_2D_points(X, y, colors, markers):
    if y is not None and len(y) == len(X):
        for i, label in enumerate(np.unique(y)):
            indices = y == label
            plt.scatter(X[indices][:, 0], X[indices][:, 1], c=colors[i], marker=markers[i])
    else:
        plt.scatter(X[:, 0], X[:, 1])


def _plot_3D_points(X, y, colors, markers, fig):
    ax = fig.add_subplot(projection='3d')
    if y is not None and len(y) == len(X):
        for i, label in enumerate(np.unique(y)):
            indices = y == label
            ax.scatter(X[indices][:, 0], X[indices][:, 2], X[indices][:, 1], c=colors[i], marker=markers[i])
    else:
        ax.scatter(X[:, 0], X[:, 2], X[:, 1])
    return ax
