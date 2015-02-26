import numpy as np
from math import log

from sklearn.utils.testing import assert_array_equal, assert_almost_equal, assert_equal

from sklearn.cluster.choose_nb_cluster import (
    adjacency_matrix, _one_stability_measure, stability, distortion,
    normal_distortion, gap_statistic, distortion_jump, max_silhouette,
    calc_CH_index, max_CH_index, elbow_within, elbow_between,
)
from sklearn.cluster.k_means_ import k_means
from sklearn import preprocessing


def test_adjacency_matrix():
    assi = [0, 0, 1, 1]
    adj_matrix = np.array([[1, 1, 0, 0], [1, 1, 0, 0],
                           [0, 0, 1, 1], [0, 0, 1, 1]])
    found_adj = adjacency_matrix(assi)
    assert_array_equal(found_adj, adj_matrix)


def test_one_stability_measure():
    X = np.arange(10) < 5
    X.reshape((10, 1))

    # test perfect clustering has 1 stability
    same_clusters = lambda x: x
    assert_almost_equal(_one_stability_measure(same_clusters, X, .8), 1)

    # test k means
    generator = np.random.RandomState(0)
    X = generator.uniform(size=(50, 2))
    clu_meth = lambda X: k_means(X, 2, random_state=0)[1]
    assert_almost_equal(_one_stability_measure(clu_meth, X, .8, 0), 0.603658536)


def test_stability():
    generator = np.random.RandomState(0)
    # for j in [20 * i: 20 * (i+1)[, x[j] = [rand rand] + [4 * i, 4 * i]
    X = generator.uniform(size=(60, 2))
    offset = np.dot(np.arange(60).reshape((60, 1)) / 20 * 1, np.ones((1, 2)))
    X += offset
    clu_meth = lambda X, k: k_means(X, k, random_state=1)[1]
    assert_equal(stability(X, clu_meth, k_max=6, nb_draw=10, random_state=0), 3)


def test_distortion():
    X = np.array([[0, 0], [2, 2],
                  [5, 5], [6, 6]])
    labels = [0, 0, 1, 1]
    assert_equal(distortion(X, labels), 2.5)


def test_normal_distortion():
    def clu_meth(points):
        n = len(points)
        mid = n / 2
        return [int(i < mid) for i in range(n)]
    assert_almost_equal(np.mean(normal_distortion(
        np.zeros((100, 2)), clu_meth, nb_draw=10, random_state=0)),
                        0.93731154)


def test_gap_statistic():
    generator = np.random.RandomState(0)
    # for j in [20 * i: 20 * (i+1)[, x[j] = [rand rand] + [4 * i, 4 * i]
    X = generator.uniform(size=(100, 2))
    offset = np.dot(np.arange(100).reshape((100, 1)) / 20 * 4, np.ones((1, 2)))
    X += offset
    clu_meth = lambda X, k: k_means(X, k, random_state=1)[1]
    assert_equal(gap_statistic(X, clu_meth, k_max=10, nb_draw=10, random_state=0), 5)
    assert_equal(gap_statistic(X, clu_meth, k_max=10, nb_draw=10, random_state=0,
                               draw_model='normal'), 5)


def test_distortion_jump():
    generator = np.random.RandomState(0)
    # for j in [20 * i: 20 * (i+1)[, x[j] = [rand rand] + [4 * i, 4 * i]
    X = generator.uniform(size=(60, 2))
    offset = np.dot(np.arange(60).reshape((60, 1)) / 20 * 1, np.ones((1, 2)))
    X += offset
    clu_meth = lambda X, k: k_means(X, k, random_state=1)[1]
    assert_equal(distortion_jump(X, clu_meth, k_max=6), 3)


def test_silhouette():
    generator = np.random.RandomState(0)
    # for j in [20 * i: 20 * (i+1)[, x[j] = [rand rand] + [4 * i, 4 * i]
    X = generator.uniform(size=(100, 2))
    offset = np.dot(np.arange(100).reshape((100, 1)) / 20 * 1, np.ones((1, 2)))
    X += offset
    clu_meth = lambda X, k: k_means(X, k, random_state=1)[1]
    assert_equal(max_silhouette(X, clu_meth, k_max=6), 5)


def test_calc_CH():
    X = np.array([[0, 0]] * 50 + [[1, 1]] * 50
                 + [[3, 3]] * 50 + [[4, 4]] * 50)
    labels = [0] * 100 + [1] * 100
    assert_equal(calc_CH_index(X, labels), 4.5 * (200 - 2) / .5) 


def test_Calinsk_Harabasz():
    generator = np.random.RandomState(0)
    # for j in [20 * i: 20 * (i+1)[, x[j] = [rand rand] + [4 * i, 4 * i]
    X = generator.uniform(size=(100, 2))
    offset = np.dot(np.arange(100).reshape((100, 1)) / 20 * 1, np.ones((1, 2)))
    X += offset
    clu_meth = lambda X, k: k_means(X, k, random_state=1)[1]
    assert_equal(max_CH_index(X, clu_meth, k_max=6), 5)


def test_elbow_within():
    generator = np.random.RandomState(0)
    # for j in [20 * i: 20 * (i+1)[, x[j] = [rand rand] + [4 * i, 4 * i]
    X = generator.uniform(size=(100, 2))
    offset = np.dot(np.arange(100).reshape((100, 1)) / 20 * 1, np.ones((1, 2)))
    X += offset
    clu_meth = lambda X, k: k_means(X, k, random_state=1)[1]
    assert_equal(elbow_within(X, clu_meth, k_max=6), 3)


def test_elbow_between():
    generator = np.random.RandomState(0)
    # for j in [20 * i: 20 * (i+1)[, x[j] = [rand rand] + [4 * i, 4 * i]
    X = generator.uniform(size=(100, 2))
    offset = np.dot(np.arange(100).reshape((100, 1)) / 20 * 1, np.ones((1, 2)))
    X += offset
    clu_meth = lambda X, k: k_means(X, k, random_state=1)[1]
    assert_equal(elbow_between(X, clu_meth, k_max=6), 2)
