import numpy as np


def _get_markers(count: int = 1, random: bool = False):
    """
    Get list of markers which can be used during plotting
    charts or diagrams in matplotlib package.

    Parameters
    ----------
    n : int {default: 1}
        Number of needed marker. Has to be >=1.

    random : bool {default: False}
        Are markers have to be given randomly.

    References
    ----------
    [1] https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    Returns
    -------
    markers
        List of markers ready to use in matplotlib package.
    """
    markers = np.array(['o', 'v', '^', '<', '>',
                        '1', '2', '3', '4', '8',
                        's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X'])

    if random:
        np.random.shuffle(markers)

    return markers[:count]


def _get_colors(count: int = 1, kind: str = "float", random: bool = False):
    """
    Get list of colors which can be used during plotting
    charts or diagrams in matplotlib package

    Parameters
    ----------
    count : int {default: 1}
        number of needed marker. Has to be >=1.

    kind : str {default: "float"}
        Data type (in string) whose colors will be returned.

    random : bool {default: False}
        Are colors have to be given randomly.

    Returns
    -------
    colors
        List of colors ready to use in matplotlib package
    """
    colors_str = ["blue", "orange", "green", "red", "purple", "brown",
                  "pink", "gray", "olive", "cyan", "rosybrown", "goldenrod",
                  "aquamarine", "darkslategrey", "skyblue", "magenta",
                  "indigo", "crimson"]

    if kind in ["int", "float"]:
        colors = np.linspace(0, 1, count)
    elif kind in ["str"]:
        colors = colors_str[:count]
    else:
        colors = None

    if random and colors is not None:
        np.random.shuffle(colors)

    return colors
