from __future__ import division
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine

from ..utils import check_random_state


def stability(X, cluster_method, k_max=None, nb_draw=100, prop_subset=.8,
              random_state=None):
    """Stability algorithm.
    For k from 2 to k_max, compute stability of cluster method to produce k
    clusters. Stability measures if the method produces the same clusters
    given small variations in the input data. It draws two overlapping subsets
    A and B of input data. For points in the two subsets, we compute the
    connectivity matrix M_A and M_B for the clustering done on subsets A and B.
    The stability of cluster_method with k cluster is the expectation of
    <M_A, M_B> / ||M_A|| * ||M_B||

    Ref: Ben-Hur, Elisseeff, Guyon: a stability based method for discovering
    structure in clusterd data, 2002
    Overview of stability: Luxburg: clustering stability: an overview

    Parameters
    ----------
    X : array-like or sparse matrix, shape (n_samples, n_features)
        The observations to cluster.

    cluster_method: function X, k (int) -> array (n_samples, 1)
        function that assign each row of X to its cluster 0 <= i < k

    k_max: int: maximum number of clusters (default = n_samples / 2)

    nb_draw: number of draws to estimate expectation of
        <M_A, M_B> / ||M_A|| * ||M_B||

    prop_subset: 0 < float < 1: proportion of input data taken in each subset

    Return
    ------
    k: int
    """
    rng = check_random_state(random_state)

    n_samples, n_features = X.shape
    if not k_max:
        k_max = n_samples / 2

    best_stab, best_k = 0, 0
    for k in range(2, k_max):
        clu_meth = lambda X: cluster_method(X, k)
        this_score = sum(_one_stability_measure(clu_meth, X, prop_subset)
                         for _ in range(nb_draw)) / nb_draw

        if this_score >= best_stab:
            best_stab = this_score
            best_k = k

    return best_k


def adjacency_matrix(cluster_assignement):
    """
    Parameter
    ---------
    cluster_assignement: vector (n_samples) of int i, 0 <= i < k

    Return
    ------
    adj_matrix: matrix (n_samples, n_samples)
        adji_matrix[i, j] = cluster_assignement[i] == cluster_assignement[j]
    """
    n_samples = len(cluster_assignement)
    adj_matrix = np.zeros((n_samples, n_samples))
    for i, val in enumerate(cluster_assignement):
        for j in range(i, n_samples):
            linked = val == cluster_assignement[j]
            adj_matrix[i, j] = linked
            adj_matrix[j, i] = linked
    return adj_matrix


def _one_stability_measure(cluster_method, X, prop_sample, random_state=None):
    """
    Draws two subsets A and B from X, apply clustering and return
    <M_A, M_B> / ||M_A|| * ||M_B||

    Parameter
    ---------
    X: array of size n_samples, n_features
    cluster_method: func X -> array (n_samples). Assignement of each point of X
        to a cluster
    prop_sample: 0 < float < 1, proportion of X taken in each subset
    """
    rng = check_random_state(random_state)

    n_sample = X.shape[0]
    set_1 = rng.uniform(size=n_sample) < prop_sample
    set_2 = rng.uniform(size=n_sample) < prop_sample
    nb_points_1, nb_points_2 = 0, 0
    points_1, points_2 = [], []
    common_points_1, common_points_2 = [], []
    for i, (is_1, is_2) in enumerate(zip(set_1, set_2)):
        if is_1 and is_2:
            common_points_1.append(nb_points_1)
            common_points_2.append(nb_points_2)
        if is_1:
            points_1.append(i)
            nb_points_1 += 1
        if is_2:
            points_2.append(i)
            nb_points_2 += 1

    assi_1 = cluster_method(X[np.ix_(points_1)])
    assi_2 = cluster_method(X[np.ix_(points_2)])

    adj_mat_1 = adjacency_matrix(assi_1)[np.ix_(common_points_1,
                                                common_points_1)]
    adj_mat_2 = adjacency_matrix(assi_2)[np.ix_(common_points_2,
                                                common_points_2)]
    return 1 - cosine(adj_mat_1.flatten(), adj_mat_2.flatten())


def distortion(X, labels):
    """
    Given data and their cluster assigment, compute the distortion D

    D = \sum_{x \in X}||x - c_x||^2

    With c_x the center of the cluster containing x
    """
    assi = defaultdict(list)
    for i, l in enumerate(labels):
        assi[l].append(i)

    centers = {lab: np.mean(X[point, :], axis=0)
               for lab, point in assi.items()}

    inertia = .0
    for x, lab in zip(X, labels):
        inertia += np.sum((x - centers[lab]) ** 2)

    return inertia / X.shape[1]
