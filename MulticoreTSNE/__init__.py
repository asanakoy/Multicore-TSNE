from __future__ import print_function
from glob import glob
import threading
import os
import sys

import numpy as np
import cffi

'''
    Helper class to execute TSNE in separate thread.
'''


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        self._target(*self._args)


class MulticoreTSNE:
    """
    Compute t-SNE embedding using Barnes-Hut optimization and
    multiple cores (if available).

    Parameters mostly correspond to parameters of `sklearn.manifold.TSNE`.

    The following parameters are unused:
    * n_iter_without_progress
    * min_grad_norm
    * method

    Args:
    cheat_metric: if true squared equclidean distance is used to build VPTree.
        Usually leads to the same quality, but much faster.
    should_normalize_input: if true normalize input features to zero mean
        and rescale values in each column to have max_value=1.
    is_frozen_point: None or np.array of bool.
        If init is a precomputed array then np.array of bool can be passed to
        define which embedding points must be frozen and not updated during optimization.
        If None - update all points.

    Parameter `init` doesn't support 'pca' initialization, but a precomputed
    array can be passed.
    """
    def __init__(self,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=12,
                 learning_rate=200,
                 n_iter=1000,
                 n_iter_without_progress=30,
                 min_grad_norm=1e-07,
                 metric='euclidean',
                 init='random',
                 is_frozen_init=None,
                 verbose=0,
                 random_state=None,
                 method='barnes_hut',
                 angle=0.5,
                 should_normalize_input=True,
                 n_jobs=1):
        self.n_components = n_components
        self.angle = angle
        self.should_normalize_input = should_normalize_input
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = -1 if random_state is None else random_state
        self.metric = metric
        self.init = init
        self.is_frozen_init = is_frozen_init
        self.embedding_ = None
        self.n_iter_ = None
        self.kl_divergence_ = None
        self.verbose = int(verbose)
        assert method == 'barnes_hut', 'Only Barnes-Hut method is allowed'
        assert isinstance(init, np.ndarray) or init == 'random', "init must be 'random' or array"
        if isinstance(init, np.ndarray):
            assert init.ndim == 2, "init array must be 2D"
            assert init.shape[1] == n_components, "init array must be of shape (n_instances, n_components)"
            self.init = np.ascontiguousarray(init, float)
            if isinstance(is_frozen_init, np.ndarray) and is_frozen_init.dtype == 'bool':
                if len(is_frozen_init) != len(init):
                    raise ValueError('is_frozen_init must be a boolean np.array '
                                     'with the same number of elements as in init, '
                                     '{} != {}'.format(len(is_frozen_init), len(init)))
            elif is_frozen_init is not None:
                raise ValueError('is_frozen_init must be None or a boolean np.array'
                                 'when init != "random"')
        elif init != 'random':
            raise ValueError('init must be "random" or np.array')
        elif is_frozen_init is not None:
            raise ValueError('is_frozen_init must be None if init = "random". '
                             'We cannot freeze random points.')

        self.ffi = cffi.FFI()
        self.ffi.cdef(
            """void tsne_run_double(double* X, int N, int D, double* Y,
                                    int no_dims, double perplexity, double theta,
                                    int num_threads, int max_iter, int random_state,
                                    bool init_from_Y, bool* is_frozen_Y, int verbose,
                                    double early_exaggeration, double learning_rate,
                                    double *final_error, char* metric,
                                    bool should_normalize_input);""")

        path = os.path.dirname(os.path.realpath(__file__))
        try:
            sofile = (glob(os.path.join(path, 'libtsne*.so')) +
                      glob(os.path.join(path, '*tsne*.dll')))[0]
            self.C = self.ffi.dlopen(os.path.join(path, sofile))
        except (IndexError, OSError) as e:
            print(e)
            raise RuntimeError('Cannot find/open tsne_multicore shared library')

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, _y=None):

        assert X.ndim == 2, 'X should be 2D array.'

        if self.metric.endswith('_prenormed') and not self.metric.endswith('time_prenormed'):
            norms = (X**2).sum(axis=1)
            if not np.allclose(norms, 1.0):
                raise ValueError('each row of X must have norm 1')

        # X may be modified, make a copy
        X = np.array(X, dtype=float, order='C', copy=True)

        N, D = X.shape
        init_from_Y = isinstance(self.init, np.ndarray)
        if init_from_Y:
            Y = self.init.copy('C')
            assert X.shape[0] == Y.shape[0], "n_instances in init array and X must match"
        else:
            Y = np.zeros((N, self.n_components))

        cffi_X = self.ffi.cast('double*', X.ctypes.data)
        cffi_Y = self.ffi.cast('double*', Y.ctypes.data)
        final_error = np.array(0, dtype=float)
        cffi_final_error = self.ffi.cast('double*', final_error.ctypes.data)
        cffi_metric = self.ffi.new('char[]', self.metric.encode('ascii'))

        if self.is_frozen_init is not None:
            is_frozen_Y = self.is_frozen_init.copy('C')
            cffi_is_frozen_Y = self.ffi.cast('bool*', is_frozen_Y.ctypes.data)
        else:
            cffi_is_frozen_Y = self.ffi.cast('bool*', 0)

        t = FuncThread(self.C.tsne_run_double,
                       cffi_X, N, D,
                       cffi_Y, self.n_components,
                       self.perplexity, self.angle, self.n_jobs, self.n_iter, self.random_state,
                       init_from_Y, cffi_is_frozen_Y, self.verbose, self.early_exaggeration,
                       self.learning_rate,
                       cffi_final_error, cffi_metric, self.should_normalize_input)
        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        self.embedding_ = Y
        self.kl_divergence_ = final_error
        self.n_iter_ = self.n_iter

        return Y
