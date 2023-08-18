"""
Helper functions for (pre-)processing

Authored by Eric Easthope
"""

import numpy as np

# Get N logarithmically-scaled frequency bands
def log_scale_bands(low, high, N):
    return np.around(
        np.array(
            list(
                zip(
                    np.geomspace(low, high, N + 1).tolist(),
                    np.geomspace(low, high, N + 1).tolist()[1:],
                )
            )
        ),
        3,
    ).tolist()


# Get overlap of two intervals a, b
def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


# Merge invalid baseline overlaps
# https://stackoverflow.com/a/5679899/18372312
def merge(times):
    saved = list(times[0])
    for st, en in sorted([sorted(t) for t in times]):
        if st <= saved[1]:
            saved[1] = max(saved[1], en)
        else:
            yield tuple(saved)
            saved[0] = st
            saved[1] = en
    yield tuple(saved)


# https://stackoverflow.com/a/9245943
def neighbours(xmax, ymax, x, y):
    """The x- and y- components for a single cell in an eight connected grid

    Parameters
    ----------
    xmax : int
        The width of the grid

    ymax: int
        The height of the grid

    x : int
        The x- position of cell to find neighbours of

    y : int
        The y- position of cell to find neighbours of

    Returns
    -------
    results : list of tuple
        A list of (x, y) indices for the neighbours
    """
    results = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            newx = x + dx
            newy = y + dy
            if dx == 0 and dy == 0:
                continue
            if newx >= 0 and newx < xmax and newy >= 0 and newy < ymax:
                results.append((newx, newy))
    return results
