import numpy as np
import pandas as pd


def individual_set_size(df: pd.DataFrame) -> list:
    return [len(df[df.iloc[:, i] == 1]) for i in range(len(df.columns))]


def possible_intersections(n_sets: int) -> tuple:
    n_ttc = 2 ** n_sets
    ttc_dec = np.arange(n_ttc)
    ttc_str = [np.binary_repr(i, width=n_sets) for i in ttc_dec]
    ttc_bin = np.array([np.fromiter(x, dtype=int) for x in ttc_str])

    return ttc_str, ttc_bin


def intersecting_set_size(df: pd.DataFrame) -> list:
    intersection_sizes = list()
    _, intersections = possible_intersections(len(df.columns))

    for x in intersections:
        temp_df = df
        for i, v in enumerate(x):
            temp_df = temp_df[temp_df.iloc[:, i] == v]
        
        intersection_sizes.append(len(temp_df))

    return intersection_sizes


def get_nodes_and_edges(n_sets: int):
    n_ttc = 2 ** n_sets
    ttc_dec = np.arange(n_ttc)
    ttc_str = [np.binary_repr(i, width=n_sets) for i in ttc_dec]
    ttc_bin = np.array([np.fromiter(x, dtype=int) for x in ttc_str])

    ones = np.ones_like(ttc_bin)

    # Node Calculation - X
    x = np.array([ones[idx] * idx for idx in range(ones.shape[0])])
    xt = x * ttc_bin
    xt = np.array([x[x != 0] for x in xt.T])
    xt = xt.T
    xf = (n_ttc - 1) - xt

    # Node Calculation - Y
    y = np.array([np.arange(1, ones.shape[1] + 1, 1)] * ones.shape[0])
    yt = y * ttc_bin
    yt = np.array([x[x != 0] for x in yt.T])
    yt = yt.T

    # Node Calculation - [X, Y]
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

    # Edge Calculation
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
    
    # Node Calculation - Y assignment to X
    yt = np.array([np.arange(0, n_sets) for _ in range(xt.shape[0])])
    yf = np.array([np.arange(0, n_sets) for _ in range(xf.shape[0])])
    
    return (xt, yt), (xf, yf), edges


def get_nonzero_nodes_and_edges(t: tuple, f: tuple, edges: list, nonzero_indices: list):
    xt, yt = t
    xf, yf = f

    # Non-Zero True Nodes
    _xt, _yt = list(), list()
    for x, y in zip(xt, yt):
        temp_x, temp_y = list(), list()
        for _x, _y in zip(x, y):
            if _x in nonzero_indices:
                p = np.where(nonzero_indices == _x)
                p = np.squeeze(p)[()]
                temp_x.append(p)
                temp_y.append(_y)
        if len(temp_x) != 0 and len(temp_y) != 0:
            _xt.append(temp_x)
            _yt.append(temp_y)

    # Non-Zero False Nodes
    _xf, _yf = list(), list()
    for x, y in zip(xf, yf):
        temp_x, temp_y = list(), list()
        for _x, _y in zip(x, y):
            if _x in nonzero_indices:
                p = np.where(nonzero_indices == _x)
                p = np.squeeze(p)[()]
                temp_x.append(p)
                temp_y.append(_y)
        if len(temp_x) != 0 and len(temp_y) != 0:
            _xf.append(temp_x)
            _yf.append(temp_y)

    # Non-Zero Edges
    _edges = list()
    for e in edges:
        if e[0][0] in nonzero_indices:
            p = np.where(nonzero_indices == e[0][0])
            p = np.squeeze(p)[()]
            temp_e = np.array(e).T
            temp_e[0] = [p] * 2
            temp_e = np.array(temp_e).T
            temp_e = temp_e.tolist()
            _edges.append(temp_e)

    return (_xt, _yt), (_xf, _yf), _edges


def get_sorted_nodes_and_edges(t: tuple, f: tuple, edges: list, sorted_sequence: list):
    xt, yt = t
    xf, yf = f

    _xt = np.concatenate(xt, axis=None)
    _xt = np.array([np.squeeze(np.where(sorted_sequence == x))[()] for x in _xt])
    _xt = _xt.reshape(xt.shape)

    _xf = np.concatenate(xf, axis=None)
    _xf = np.array([np.squeeze(np.where(sorted_sequence == x))[()] for x in _xf])
    _xf = _xf.reshape(xf.shape)

    #  Sorted Edges
    _edges = list()
    for e in edges:
        if e[0][0] in sorted_sequence:
            p = np.where(sorted_sequence == e[0][0])
            p = np.squeeze(p)[()]
            temp_e = np.array(e).T
            temp_e[0] = [p] * 2
            temp_e = np.array(temp_e).T
            temp_e = temp_e.tolist()
            _edges.append(temp_e)

    return (_xt, yt), (_xf, yf), _edges
