# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pathos.multiprocessing import ProcessingPool as Pool

from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class DistanceMatrix(object):
    """A distance matrix.

    Args:
        path_or_array_like (str or array-like): If str, path to csv
            containing distance matrix. If array-like, the distance matrix.
            Must be square.
        na_values (str): How nan is represented in csv file.

    Notes:
        Negative distances are converted to 0.
    """

    def __init__(self, path_or_array_like, na_values='*'):
        if type(path_or_array_like) is str:
            df = pd.read_csv(
                path_or_array_like, index_col=0, na_values=na_values)
            self.D = df.values
            self.index = df.index

        else:
            if path_or_array_like.ndim != 2:
                raise ValueError('Data not 2 dimensional')
            if path_or_array_like.shape[1] != path_or_array_like.shape[0]:
                raise ValueError('Data must be square')

            if type(path_or_array_like) is pd.DataFrame:
                self.D = path_or_array_like.values
                self.index = path_or_array_like.index
            else:
                self.D = path_or_array_like

        self.D[self.D < 0] = 0
        self.m = self.D.shape[0]

    def _error(self, diff):
        """Sum of the squared difference.

        Args:
            diff (array-like): [m, m] matrix.

        Returns:
            (float)
        """
        return np.nansum(np.power(diff, 2)) / 2

    def _gradient(self, diff, d, coords):
        """Compute the gradient.

        Args:
            diff (array-like): [m, m] matrix. D - d
            d (array-like): [m, m] matrix.
            coords (array-like): [m, n] matrix.

        Returns:
            (np.array) [m, n]
        """
        denom = np.copy(d)
        denom[denom == 0] = 1e-5

        with np.errstate(divide='ignore', invalid='ignore'):
            K = -2 * diff / denom

        K[np.isnan(K)] = 0

        g = np.empty_like(coords)
        for n in range(self.n):
            for i in range(self.m):
                # Vectorised version of (~70 times faster)
                # for j in range(self.m):
                #     delta_g = ((coords[i, n] - coords[j, n]) * K[i, j]).sum()
                #     g[i, n] += delta_g
                g[i, n] = ((coords[i, n] - coords[:, n]) * K[i, :]).sum()

        return g

    def _error_and_gradient(self, x):
        """Compute the error and the gradient.

        This is the function optimised by :obj:`scipy.optimize.minimize`.

        Args:
            x (array-like): [m * n, ] matrix.

        Returns:
            (tuple) containing

                - (float)
                - (np.array)
        """
        coords = x.reshape((self.m, self.n))
        d = squareform(pdist(coords))
        diff = self.D - d
        error = self._error(diff)
        gradient = self._gradient(diff, d, coords)
        return error, gradient.ravel()

    def optimise(self, start=None, n=2):
        """Run multidimensional scaling on this distance matrix.

        Args:
            start (None or array-like): Starting coordinates. If None, random
                starting coordinates are used. If array-like must have shape
                [m * n, ].
            n (int): Number of dimensions to embed samples in.

        Examples:

            .. doctest::

               >>> import pandas as pd
               >>> from pymds.mds import DistanceMatrix
               >>> dist = pd.DataFrame({
               ...    'a': [0.0, 1.0, 2.0],
               ...    'b': [1.0, 0.0, 3 ** 0.5],
               ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
               >>> dm = DistanceMatrix(dist)
               >>> proj = dm.optimise(n=2)
               >>> proj.coords.shape
               (3, 2)
               >>> type(proj)
               pymds.mds.Projection


        Returns:
            (Projection) The multidimensional scaling result.
        """
        self.n = n

        start = start or np.random.rand(self.m * self.n) * 10

        optim = minimize(
            fun=self._error_and_gradient,
            x0=start,
            jac=True,
            method='L-BFGS-B')

        index = self.index if hasattr(self, "index") else None

        return Projection(optim, n=self.n, m=self.m, index=index)

    def optimise_batch(self, batchsize=10, returns='best', paralell=True):
        """
        Run multiple optimisations using different starting coordinates.

        Args:
            m (int): Number of dimensions to embed samples in.
            batchsize (int): Number of optimisations to run.
            returns (str): If 'all', return results of all optimisations. These
                are ordered by stress, ascending. If 'best' return only one
                Projection with the lowest stress.
            parallel (bool): Run optimisations in parallel or not.

        Examples:

            .. doctest::

               >>> import pandas as pd
               >>> from pymds.mds import DistanceMatrix
               >>> dist = pd.DataFrame({
               ...    'a': [0.0, 1.0, 2.0],
               ...    'b': [1.0, 0.0, 3 ** 0.5],
               ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
               >>> dm = DistanceMatrix(dist)
               >>> batch = dm.optimise_batch(batchsize=3, returns='all')
               >>> len(batch)
               3
               >>> type(batch[0])
               pymds.mds.Projection

        Returns:
            (list) of length batchsize. Contains instances of (Projection).
                Sorted by stress, ascending.

            or

            (Projection) with the lowest stress.
        """
        if returns not in ('best', 'all'):
            raise ValueError('returns must be either "best" or "all"')

        starts = [np.random.rand(self.m * 2) * 10 for i in range(batchsize)]

        if paralell:
            with Pool() as p:
                results = p.map(self.optimise, starts)
        else:
            results = map(self.optimise, starts)

        results = sorted(results, key=lambda x: x.stress)

        return results if returns == 'all' else results[0]


class Projection(object):
    """Samples embeded in n-dimensional space.

    Args:
        OptimizeResult (scipy.optimize.OptimizeResult): Object returned by
            `scipy.optimize.minimize`.
        n (int): Number of dimensions.
        m (int): Number of samples.
        index (list-like): Names of samples. (Optional).

    Attributes:
        coords (pd.DataFrame): Coordinates of the projection.
        stress (float): Residual error of multidimensional scaling.
    """

    def __init__(self, OptimizeResult, n, m, index=None):
        self.coords = pd.DataFrame(
            OptimizeResult.x.reshape((m, n)), index=index)
        self.stress = OptimizeResult.fun

    def plot(self, **kwargs):
        """Plots the coordinates of the first two dimensions of the projection.

        Removes all axis and tick labels, and sets the grid spacing at 1 unit.
        One way to display the grid, using seaborn, is:

        Examples:

            .. doctest::

               >>> import pandas as pd
               >>> # Seaborn used for styles
               >>> import seaborn as sns
               >>> sns.set_style('whitegrid')
               >>> from pymds.mds import DistanceMatrix
               >>> dist = pd.DataFrame({
               ...    'a': [0.0, 1.0, 2.0],
               ...    'b': [1.0, 0.0, 3 ** 0.5],
               ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
               >>> dm = DistanceMatrix(dist)
               >>> proj = dm.optimise()
               >>> proj.plot()

        Args:
            kwargs (dict): Passed to `pd.DataFrame.plot.scatter`.

        Returns:
            (matplotlib.axes.Subplot)
        """
        self.coords.plot.scatter(x=0, y=1, **kwargs)
        ax = plt.gca()
        ax.get_xaxis().set_major_locator(MultipleLocator(base=1.0))
        ax.get_yaxis().set_major_locator(MultipleLocator(base=1.0))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect(1)
        return ax
