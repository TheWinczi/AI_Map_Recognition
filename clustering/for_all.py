import numpy as np
from plotting import plot_clusterizers_comparison, plot_points
from .kmeans import kmeans
from .agglomerative_clustering import agg
from .spectral_clustering import spectral_clustering
from .dbscan import dbscan
from .optics import optics


def clusterize(X: np.ndarray):
    """
    Clusterize the input data.

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of tuples/lists in shape
        [(x_0, y_0), (x_1, y_1), ...].

    Returns
    -------
    y_cls
        List of labels belongs to input X data.
    """
    agg_ = agg(X)
    return agg_.labels_


def try_all_clusterizers(X: np.ndarray, compare: bool = True):
    """
    Try all clustering algorithms which was implemented to clusterize input data.

    Parameters
    ----------
    X : ndarray
        Array of points coordinates as a list of tuple/lists in shape
        [(x_0, y_0), (x_1, y_1), ...].

    compare : bool {default: True}
        Are all algorithms results have to be compared on chart.

    Returns
    -------
    y_cls
        Array of labels assigned to input X data.
    """
    kmeans_ = kmeans(X)
    plot_points(X, kmeans_.labels_,
                title="KMeans clustering results", xlabel="x coord", ylabel="y coord")
    agg_ = agg(X)
    plot_points(X, agg_.labels_,
                title="AgglomerativeClustering clustering results", xlabel="x coord", ylabel="y coord")
    spec_ = spectral_clustering(X)
    plot_points(X, spec_.labels_,
                title="SpectralClustering clustering results", xlabel="x coord", ylabel="y coord")
    dbscan_ = dbscan(X)
    plot_points(X, dbscan_.labels_,
                title="DBSCAN clustering results", xlabel="x coord", ylabel="y coord")
    optics_ = optics(X)
    plot_points(X, optics_.labels_,
                title="OPTICS clustering results", xlabel="x coord", ylabel="y coord")

    # KMeans, AgglomerativeClustering and SpectralClustering clustering algorithms gets satisfying result.
    if compare:
        plot_clusterizers_comparison([kmeans_, agg_, spec_],
                                     X=X,
                                     xlabels=["x_axis" for _ in range(3)],
                                     ylabels=["y_axis" for _ in range(3)],
                                     titles=["KMeans", "AgglomerativeClustering", "SpectralClustering"],
                                     suptitle="The best clustering algorithms comparison")
    return spec_.labels_
