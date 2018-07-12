# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pathos.multiprocessing import ProcessingPool as Pool

from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection


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
        """Half the sum of the squared difference.

        Args:
            diff (array-like): [m, m] matrix.

        Returns:
            (float)

        Notes:
            The return value is divided by two because distances between each
            pair of samples are represented twice in a full [m, m] matrix.
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

        This is the function optimized by :obj:`scipy.optimize.minimize`.

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

    def optimize(self, start=None, n=2):
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
               >>> pro = dm.optimize(n=2)
               >>> pro.coords.shape
               (3, 2)
               >>> type(proj)
               <class 'pymds.mds.Projection'>


        Returns:
            (pymds.Projection) The multidimensional scaling result.
        """
        self.n = n

        if start is None:
            start = np.random.rand(self.m * self.n) * 10

        optim = minimize(
            fun=self._error_and_gradient,
            x0=start,
            jac=True,
            method='L-BFGS-B')

        index = self.index if hasattr(self, "index") else None

        return Projection.from_optimize_result(
            OptimizeResult=optim, n=self.n, m=self.m, index=index)

    def optimize_batch(self, batchsize=10, returns='best', paralell=True):
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
               >>> batch = dm.optimize_batch(batchsize=3, returns='all')
               >>> len(batch)
               3
               >>> type(batch[0])
               <class 'pymds.mds.Projection'>

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
                results = p.map(self.optimize, starts)
        else:
            results = map(self.optimize, starts)

        results = sorted(results, key=lambda x: x.stress)

        return results if returns == 'all' else results[0]


class Projection(object):
    """Samples embeded in n-dimensional space.

    Args:
        coords (pandas.DataFrame): Coordinates of the projection.

    Attributes:
        coords (pandas.DataFrame): Coordinates of the projection.
        stress (float): Residual error of multidimensional scaling. (If
            generated using `self.from_optimize_result`).
    """

    def __init__(self, coords):
        self.coords = pd.DataFrame(coords)

    @classmethod
    def from_optimize_result(cls, OptimizeResult, n, m, index=None):
        """Construct a Projection from the output of an optimization.

        Args:
            OptimizeResult (`scipy.optimize.OptimizeResult`): Object returned
                by `scipy.optimize.minimize`.
            n (int): Number of dimensions.
            m (int): Number of samples.
            index (list-like): Names of samples. (Optional).

        Returns:
            (pymds.Projection)
        """
        coords = pd.DataFrame(OptimizeResult.x.reshape((m, n)), index=index)
        projection = cls(coords)
        projection.stress = OptimizeResult.fun
        return projection

    def plot(self, **kwargs):
        """Plots the coordinates of the first two dimensions of the projection.

        Removes all axis and tick labels, and sets the grid spacing at 1 unit.
        One way to display the grid, using seaborn, is:

        Examples:

                >>> import pandas as pd
                >>> from pymds.mds import DistanceMatrix
                ...
                >>> # Seaborn used for styles
                >>> import seaborn as sns
                >>> sns.set_style('whitegrid')
                ...
                >>> dist = pd.DataFrame({
                ...    'a': [0.0, 1.0, 2.0],
                ...    'b': [1.0, 0.0, 3 ** 0.5],
                ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
                >>> dm = DistanceMatrix(dist)
                >>> pro = dm.optimize()
                >>> pro.plot()

        Args:
            kwargs: Passed to `pd.DataFrame.plot.scatter`.

        Returns:
            (matplotlib.axes.Subplot)
        """
        ax = plt.gca()
        self.coords.plot.scatter(x=0, y=1, ax=ax, **kwargs)
        ax.get_xaxis().set_major_locator(MultipleLocator(base=1.0))
        ax.get_yaxis().set_major_locator(MultipleLocator(base=1.0))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect(1)
        return ax

    def plot_lines_to(self, other, index=None, **kwargs):
        """Plot lines from samples shared between this projection and another
        dataset.

        Args:
            other (`pymds.Projection` or `pandas.DataFrame` or `array-like`):
                The other dataset to plot lines to. If other is an
                instance of `pymds.Projection` or `pandas.DataFrame`, then
                other must have indexes in common with this projection. If
                `array-like`, then other must have the same dimensions as
                `self.coords`.
            index (`list-like` or `None`): Only draw lines between samples in
                index. All elements in index must be samples in this projection
                and other.
            kwargs: Passed to `matplotlib.collections.LineCollection`. Useful
                keywords include `linewidths`, `colors` and `zorder`.

        Examples:

                >>> import numpy as np
                >>> from pymds.mds import Projection
                ...
                >>> pro = Projection(np.random.randn(50, 2))
                ... 
                >>> # Rotate projection 90 deg
                >>> R = np.array([[0, -1], [1, 0]])
                >>> other = np.dot(pro.coords, R)
                ... 
                >>> projection.plot(c='black', edgecolor='white', zorder=20)
                >>> projection.plot_lines_to(
                ...     other, linewidths=0.3, colors='darkgrey')


        Returns:
            (matplotlib.axes.Subplot)
        """
        is_projection = type(other) is Projection
        is_df = type(other) is pd.DataFrame

        if is_projection or is_df:
            df_other = other.coords if is_projection else other

            if index is not None:

                uniq_idx = set(index)

                if uniq_idx - set(df_other.index):
                    raise ValueError(
                        "Samples in index are not in other")

                if uniq_idx - set(self.coords.index):
                    raise ValueError(
                        "Samples in index are not in this projection")

            else:
                uniq_idx = set(df_other.index) & set(self.coords.index)

                if not uniq_idx:
                    raise ValueError(
                        "This projection shares no samples with other")

            idx = list(uniq_idx)
            start = self.coords.loc[idx, :].values
            end = df_other.loc[idx, :].values

        else:
            if not hasattr(other, "ndim"):
                raise TypeError(
                    "other not array-like, or pandas.DataFrame, or "
                    "pymds.Projection")

            if other.shape != self.coords.shape:
                raise ValueError(
                    "array-like must have the same shape as self.coords")

            start = self.coords.values
            end = other

        segments = [[start[i, :], end[i, :]] for i in range(start.shape[0])]

        ax = plt.gca()
        ax.add_artist(LineCollection(segments=segments, **kwargs))
        return ax


    def orient_to(self, other, index=None, inplace=False, scaling=False):
        """Orient this Projection to another dataset.

        Orient this projection using reflection, rotation and translation to
        match another projection using procrustes superimposition. Scaling is
        optional.

        Args:
            other (pymds.Projection or pandas.DataFrame or array-like): The
                other dataset to orient this projection to. If other is an
                instance of pymds.Projection or pandas.DataFrame, then other
                must have indexes in common with this projection. If
                array-like, then other must have the same dimensions as
                self.coords.
            index (list-like or None): If other is an instance of
                pandas.DataFrame or pymds.Projection then orient this
                projection to other using only samples in index.
            inplace (bool): Either update the coordinates of this projection
                inplace, or return a new instance of pymds.Projection.
            scaling (bool): Allow scaling. (Not implemented yet).

        Examples:

            .. doctest::

                >>> import numpy as np
                >>> import pandas as pd
                >>> from pymds.mds import Projection
                ...
                >>> array = np.random.randn(10, 2)
                >>> pro = Projection(pd.DataFrame(array))
                ...
                >>> # Flip left-right, rotate 90 deg and translate
                >>> other = np.fliplr(array)
                >>> other = np.dot(other, np.array([[0, -1], [1, 0]]))
                >>> other += np.array([10, -5])
                ...
                >>> oriented = pro.orient_to(other)
                >>> (oriented.coords.values - other).sum() < 1e-6
                True

        Returns:
            (pymds.Projection): If not inplace.
        """
        is_projection = type(other) is Projection
        is_df = type(other) is pd.DataFrame

        if is_projection or is_df:
            df_other = other.coords if is_projection else other

            if index:
                # Check all indexes have no repeats
                uniq_idx = set(index)
                if len(uniq_idx) != len(index):
                    raise ValueError("index has repeat elements")

                uniq_other_idx = set(df_other.index)
                if len(uniq_other_idx) != len(df_other.index):
                    raise ValueError("other index has repeat elements")

                uniq_self_idx = set(self.coords.index)
                if len(uniq_self_idx) != len(self.coords.index):
                    raise ValueError("self.coords.index has repeat elements")

                # Check all elements in index are in df_other.index and
                # self.coords.index
                if uniq_idx - uniq_other_idx:
                    raise ValueError(
                        "index contains elements not in other index")

                if uniq_idx - uniq_self_idx:
                    raise ValueError(
                        "index contains elements not in self.coords.index")

            else:

                uniq_idx = set(df_other.index) & set(self.coords.index)

                if not len(uniq_idx):
                    raise ValueError(
                        "No samples shared between other and this projection")

            idx = list(uniq_idx)
            arr_self = self.coords.loc[idx, :]
            arr_other = df_other.loc[idx, :]

        else:
            if not hasattr(other, "ndim"):
                raise TypeError(
                    "other not array-like, or pandas.DataFrame, or "
                    "pymds.Projection")

            if other.shape != self.coords.shape:
                raise ValueError(
                    "array-like must have the same shape as self.coords")

            else:
                arr_self = self.coords.values
                arr_other = other

        if scaling:
            raise NotImplementedError()

        else:
            self_mean = arr_self.mean()
            other_mean = arr_other.mean()

            A = arr_self - self_mean
            B = arr_other - other_mean
            R, scale = orthogonal_procrustes(A, B)

            to_rotate = self.coords - self.coords.mean()
            rotated = pd.DataFrame(
                np.dot(to_rotate, R),
                index=self.coords.index)

            oriented = rotated + other_mean

        if inplace:
            self.coords = oriented
        else:
            return Projection(oriented)
