import numpy as np


def get_nodes_and_edges(n_cat: int):
    n_ttc = 2 ** n_cat
    ttc_dec = np.arange(n_ttc)
    ttc_str = [np.binary_repr(i, width=n_cat) for i in ttc_dec]
    ttc_bin = np.array([np.fromiter(x, dtype=int) for x in ttc_str])

    ones = np.ones_like(ttc_bin)

    x = np.array([ones[idx] * idx for idx in range(ones.shape[0])])
    xt = x * ttc_bin
    xt = np.array([x[x != 0] for x in xt.T])
    xt = xt.T
    xf = (n_ttc - 1) - xt

    y = np.array([np.arange(1, ones.shape[1] + 1, 1)] * ones.shape[0])
    yt = y * ttc_bin
    yt = np.array([x[x != 0] for x in yt.T])
    yt = yt.T

    xtf = xt.flatten()
    ytf = yt.flatten()

    uniques, counts = np.unique(xtf, return_counts=True)
    trim_unique = np.array([1 if x in uniques[counts > 1] else 0 for x in xtf])
    xs = xtf * trim_unique
    xs = xs[xs != 0]

    ys = ytf * trim_unique
    ys = ys[ys != 0]
    ys = ys - 1

    xy = [(x, y) for x, y in zip(xs, ys)]
    nodes = [[x, y] for (x, y) in sorted(xy)]

    edges = list()

    for i, [x, y] in enumerate(nodes):
        if i == 0:
            prev_node = [x, y]
            prev_x = x
        else:
            if prev_x == x:
                edges.append([prev_node, [x, y]])
            if prev_x != x:
                prev_x = x
            prev_node = [x, y]
    
    return xt, xf, edges
