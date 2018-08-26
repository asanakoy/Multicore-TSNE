from __future__ import print_function
from ddt import ddt, data
import unittest
from functools import partial

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

from MulticoreTSNE import MulticoreTSNE


make_blobs = partial(make_blobs, random_state=0)
MulticoreTSNE = partial(MulticoreTSNE, random_state=0)


def pdist(X):
    """Condensed pairwise distances, like scipy.spatial.distance.pdist()"""
    return pairwise_distances(X)[np.triu_indices(X.shape[0], 1)]


@ddt
class TestMulticoreTSNE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Xy = make_blobs(91, 100, 2, shuffle=False)
        cls.Xy_small = make_blobs(20, 100, 2, shuffle=False)

    def test_tsne(self):
        X, y = self.Xy_small
        tsne = MulticoreTSNE(perplexity=5, n_iter=500)
        E = tsne.fit_transform(X)

        self.assertEqual(E.shape, (X.shape[0], 2))

        max_intracluster = max(pdist(E[y == 0]).max(),
                               pdist(E[y == 1]).max())
        min_intercluster = pairwise_distances(E[y == 0],
                                              E[y == 1]).min()

        self.assertGreater(min_intercluster, max_intracluster)

    def test_n_jobs(self):
        X, y = self.Xy
        tsne = MulticoreTSNE(n_iter=100, n_jobs=2)
        tsne.fit_transform(X)

    def test_perplexity(self):
        X, y = self.Xy
        tsne = MulticoreTSNE(perplexity=X.shape[0] // 3 - 1, n_iter=100)
        tsne.fit_transform(X)

    def test_dont_change_x(self):
        X = np.random.random((20, 4))
        X_orig = X.copy()
        MulticoreTSNE(n_iter=400, perplexity=6).fit_transform(X)
        np.testing.assert_array_equal(X, X_orig)

    def test_init_from_y(self):
        X, y = self.Xy
        tsne = MulticoreTSNE(n_iter=500)
        E = tsne.fit_transform(X)

        tsne = MulticoreTSNE(n_iter=0, init=E)
        E2 = tsne.fit_transform(X)
        np.testing.assert_allclose(E, E2)

        tsne = MulticoreTSNE(n_iter=1, init=E)
        E2 = tsne.fit_transform(X)
        mean_diff = np.abs((E - E2).sum(1)).mean()
        self.assertLess(mean_diff, 30)

    def test_lr_mult_exception(self):
        X, y = self.Xy
        lr_mult = np.ones(len(X), dtype=float)
        lr_mult[:len(X) // 2] = 0.1
        with self.assertRaises(ValueError) as context:
            MulticoreTSNE(n_iter=100, lr_mult=lr_mult)
        self.assertIn('lr_mult must be None if init = "random"', str(context.exception))

    def test_lr_mult(self):
        X, y = self.Xy
        lr_mult = np.ones(len(X), dtype=float)
        lr_mult[:len(X) // 2] = 0.1

        tsne = MulticoreTSNE(n_iter=100, n_components=2, lr_mult=lr_mult, init=X[:, :2])
        E = tsne.fit_transform(X)
        self.assertFalse(np.any(np.isnan(E)))

    def test_attributes(self):
        X, y = self.Xy
        N_ITER = 200
        tsne = MulticoreTSNE(n_iter=N_ITER)
        E = tsne.fit_transform(X, y)

        self.assertIs(tsne.embedding_, E)
        self.assertGreater(tsne.kl_divergence_, 0)
        self.assertEqual(tsne.n_iter_, N_ITER)

    @data('euclidean', 'sqeuclidean', 'cosine', 'angular', 'precomputed')
    def test_metric(self, metric):
        X, y = self.Xy_small
        tsne = MulticoreTSNE(perplexity=5, n_iter=500, metric=metric)
        if metric == 'precomputed':
            dist_matrix = pairwise_distances(X, metric='euclidean')
            E = tsne.fit_transform(dist_matrix)
        else:
            E = tsne.fit_transform(X)

        self.assertEqual(E.shape, (X.shape[0], 2))

        max_intracluster = max(pdist(E[y == 0]).max(),
                               pdist(E[y == 1]).max())
        min_intercluster = pairwise_distances(E[y == 0],
                                              E[y == 1]).min()

        self.assertGreater(min_intercluster, max_intracluster)

    def test_pairs(self):
        X, y = self.Xy
        N_ITER = 400
        tsne = MulticoreTSNE(n_iter=N_ITER, contrib_cost_pairs=0.0001, n_jobs=1, verbose=100)
        E = tsne.fit_transform(X, y, pairs=np.array([[1, 60], [3, 85], [10, 80]]))
        print('kl_divergence_', tsne.kl_divergence_)
        print('pairs_error', tsne.pairs_error)

        self.assertIs(tsne.embedding_, E)
        self.assertGreater(tsne.kl_divergence_, 0)
        self.assertGreater(tsne.pairs_error, 0)
        self.assertEqual(tsne.n_iter_, N_ITER)


if __name__ == '__main__':
    unittest.main()
