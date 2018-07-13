# -*- coding: utf-8 -*-
"""
Notes:

    `m`: Number of samples. This is the number of rows and columns in `D`.
    `n`: Number of dimensions the multidimensional scaling result is embedded
        in.
    `D`: The distance matrix. Shape (`m`, `m`)
    `coords`: Coordinates of the samples embeded in `m`-dimensional space.
        Shape (`m`, `n`).
    `x`: Coordinates of the samples embeded in `m`-dimensional space. Shape
        (`m` * `n`, ).
    `d`: Pairwise distances between samples in `coords`. Shape (`m`, `m`).
    `g`: Gradient of the error function at each coordinate in `coords`. Shape
        (`m`, `n`).
    `diff`: `D` `-` `d`.
"""
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
        path_or_array_like (`str` or `array-like`): If `str`, path to csv
            containing distance matrix. If `array-like`, the distance matrix.
            Must be square.
        na_values (`str`): How nan is represented in csv file.

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
            diff (`array-like`): [`m`, `m`] matrix.

        Returns:
            `float`: The error.

        Notes:
            The return value is divided by two because distances between each
            pair of samples are represented twice in a full [`m`, `m`] matrix.
        """
        return np.nansum(np.power(diff, 2)) / 2

    def _gradient(self, diff, d, coords):
        """Compute the gradient.

        Args:
            diff (`array-like`): [`m`, `m`] matrix. `D` - `d`
            d (`array-like`): [`m`, `m`] matrix.
            coords (`array-like`): [`m`, `n`] matrix.

        Returns:
            `np.array`: Gradient, shape [`m`, `n`].
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
            x (`array-like`): [`m` * `n`, ] matrix.

        Returns:
            `tuple`: containing:

                - Error (`float`)
                - Gradient (`np.array`) [`m`, `n`]
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
            start (`None` or `array-like`): Starting coordinates. If
                `start=None`, random starting coordinates are used. If
                `array-like` must have shape [`m` * `n`, ].
            n (`int`): Number of dimensions to embed samples in.

        Examples:

            .. doctest::

               >>> import pandas as pd
               >>> from pymds import DistanceMatrix
               >>> dist = pd.DataFrame({
               ...    'a': [0.0, 1.0, 2.0],
               ...    'b': [1.0, 0.0, 3 ** 0.5],
               ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
               >>> dm = DistanceMatrix(dist)
               >>> pro = dm.optimize(n=2)
               >>> pro.coords.shape
               (3, 2)
               >>> type(pro)
               <class 'pymds.mds.Projection'>


        Returns: :py:class:`pymds.Projection`
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
            result=optim, n=self.n, m=self.m, index=index)

    def optimize_batch(self, batchsize=10, returns='best', paralell=True):
        """
        Run multiple optimizations using different starting coordinates.

        Args:
            batchsize (`int`): Number of optimizations to run.
            returns (`str`): If ``'all'``, return results of all optimizations,
                ordered by stress, ascending. If ``'best'`` return the
                projection with the lowest stress.
            parallel (`bool`): If ``True``, run optimizations in parallel.

        Examples:

            .. doctest::

               >>> import pandas as pd
               >>> from pymds import DistanceMatrix
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
            `list` or :py:class:`pymds.Projection`:

                `list`: Length batchsize, containing instances of
                :py:class:`pymds.Projection`. Sorted by stress, ascending.

                or

                :py:class:`pymds.Projection`: Projection with the lowest
                stress.
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
    """Samples embedded in n-dimensions.

    Args:
        coords (:py:class:`pandas.DataFrame`): Coordinates of the projection.

    Attributes:
        coords (:py:class:`pandas.DataFrame`): Coordinates of the projection.
        stress (`float`): Residual error of multidimensional scaling. (If
            generated using :py:meth:`Projection.from_optimize_result`).
    """

    def __init__(self, coords):
        self.coords = pd.DataFrame(coords)

    @classmethod
    def from_optimize_result(cls, result, n, m, index=None):
        """Construct a Projection from the output of an optimization.

        Args:
            result (:py:class:`scipy.optimize.OptimizeResult`): Object 
                returned by :py:func:`scipy.optimize.minimize`.
            n (`int`): Number of dimensions.
            m (`int`): Number of samples.
            index (`list-like`): Names of samples. (Optional).

        Returns:
            :py:class:`pymds.Projection`
        """
        coords = pd.DataFrame(result.x.reshape((m, n)), index=index)
        projection = cls(coords)
        projection.stress = result.fun
        return projection

    def _get_samples_shared_with(self, other, index=None):
        """Find samples shared with another dataset.

        Args:
            other
                (:py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`
                    or `array-like`):
                The other dataset. If `other` is an instance of
                :py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`,
                then `other` must have indexes in common with this projection.
                If `array-like`, then other must have same dimensions as
                `self.coords`.
            index (`list-like` or `None`): If `other` is an instance of
                :py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`
                then only return samples in index.

        Returns:
            `tuple`: containing:

                - this (`numpy.array`) Shape [`x`, `n`].
                - other (`numpy.array`) Shape [`x`, `n`].
        """
        if isinstance(other, (pd.DataFrame, Projection)):
            df_other = other.coords if isinstance(other, Projection) else other

            if len(set(df_other.index)) != len(df_other.index):
                raise ValueError("other index has duplicates")

            if len(set(self.coords.index)) != len(self.coords.index):
                raise ValueError("This projection index has duplicates")

            if index:
                uniq_idx = set(index)

                if len(uniq_idx) != len(index):
                    raise ValueError("index has has duplicates")

                if uniq_idx - set(df_other.index):
                    raise ValueError("index has samples not in other")

                if uniq_idx - set(self.coords.index):
                    raise ValueError(
                        "index has samples not in this projection")

            else:
                uniq_idx = set(df_other.index) & set(self.coords.index)

                if not len(uniq_idx):
                    raise ValueError(
                        "No samples shared between other and this projection")

            idx = list(uniq_idx)
            return self.coords.loc[idx, :].values, df_other.loc[idx, :].values

        else:
            other = np.array(other)

            if other.shape != self.coords.shape:
                raise ValueError(
                    "array-like must have the same shape as self.coords")

            return self.coords.values, other

    def plot(self, **kwds):
        """Plot the coordinates in the first two dimensions of the projection.

        Removes axis and tick labels, and sets the grid spacing to 1 unit.
        One way to display the grid is to use `Seaborn`_:

        Args:
            **kwds: Passed to :py:meth:`pandas.DataFrame.plot.scatter`.

        Examples:

            >>> from pymds import DistanceMatrix
            >>> import pandas as pd
            >>> import seaborn as sns
            >>> sns.set_style('whitegrid')
            >>> dist = pd.DataFrame({
            ...    'a': [0.0, 1.0, 2.0],
            ...    'b': [1.0, 0.0, 3 ** 0.5],
            ...    'c': [2.0, 3 ** 0.5, 0.0]} , index=['a', 'b', 'c'])
            >>> dm = DistanceMatrix(dist)
            >>> pro = dm.optimize()
            >>> ax = pro.plot(c='black', s=50, edgecolor='white')

        Returns:
            :py:obj:`matplotlib.axes.Axes`

        .. _Seaborn:
            https://seaborn.pydata.org/
        """
        ax = plt.gca()
        self.coords.plot.scatter(x=0, y=1, ax=ax, **kwds)
        ax.get_xaxis().set_major_locator(MultipleLocator(base=1.0))
        ax.get_yaxis().set_major_locator(MultipleLocator(base=1.0))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_aspect(1)
        return ax

    def plot_lines_to(self, other, index=None, **kwds):
        """Plot lines from samples shared between this projection and another
        dataset.

        Args:
            other
                (:py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`
                or `array-like`):
                The other dataset to plot lines to. If other is an instance of
                :py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`,
                then other must have indexes in common with this projection.
                If `array-like`, then other must have the same dimensions as
                `self.coords`.
            index (`list-like` or `None`): Only draw lines between samples in
                index. All elements in index must be samples in this projection
                and other.
            **kwds: Passed to :py:obj:`matplotlib.collections.LineCollection`.

        Examples:

            >>> import numpy as np
            >>> from pymds import Projection
            >>> pro = Projection(np.random.randn(50, 2))
            >>> R = np.array([[0, -1], [1, 0]])
            >>> other = np.dot(pro.coords, R)  # Rotate 90 deg
            >>> ax = pro.plot(c='black', edgecolor='white', zorder=20)
            >>> ax = pro.plot_lines_to(other, linewidths=0.3)

        Returns:
            :py:obj:`matplotlib.axes.Axes`
        """
        start, end = self._get_samples_shared_with(other, index=index)
        segments = [[start[i, :], end[i, :]] for i in range(start.shape[0])]
        ax = plt.gca()
        ax.add_artist(LineCollection(segments=segments, **kwds))
        return ax

    def orient_to(self, other, index=None, inplace=False, scaling=False):
        """Orient this Projection to another dataset.

        Orient this projection using reflection, rotation and translation to
        match another projection using procrustes superimposition. Scaling is
        optional.

        Args:
            other
                (:py:class:`pymds.Projection` or :py:class:`pandas.DataFrame`
                or `array-like`):
                The other dataset to orient this projection to.
                If other is an instance of :py:class:`pymds.Projection` or
                :py:class:`pandas.DataFrame`, then other must have indexes in
                common with this projection. If `array-like`, then other must
                have the same dimensions as self.coords.
            index (`list-like` or `None`): If other is an instance of
                :py:class:`pandas.DataFrame` or :py:class:`pymds.Projection`
                then orient this projection to other using only samples in
                index.
            inplace (`bool`): Update coordinates of this projection inplace,
                or return an instance of :py:class:`pymds.Projection`.
            scaling (`bool`): Allow scaling. (Not implemented yet).

        Examples:

            .. doctest::

                >>> import numpy as np
                >>> import pandas as pd
                >>> from pymds import Projection
                >>> array = np.random.randn(10, 2)
                >>> pro = Projection(pd.DataFrame(array))
                >>> # Flip left-right, rotate 90 deg and translate
                >>> other = np.fliplr(array)
                >>> other = np.dot(other, np.array([[0, -1], [1, 0]]))
                >>> other += np.array([10, -5])
                >>> oriented = pro.orient_to(other)
                >>> (oriented.coords.values - other).sum() < 1e-6
                True

        Returns:
            :py:class:`pymds.Projection`: If ``inplace=False``.
        """
        arr_self, arr_other = self._get_samples_shared_with(other, index=index)

        if scaling:
            raise NotImplementedError()

        else:
            self_mean = arr_self.mean()
            other_mean = arr_other.mean()

            A = arr_self - self_mean
            B = arr_other - other_mean
            R, _ = orthogonal_procrustes(A, B)

            to_rotate = self.coords - self.coords.mean()
            oriented = np.dot(to_rotate, R) + other_mean
            oriented = pd.DataFrame(oriented, index=self.coords.index)

        if inplace:
            self.coords = oriented
        else:
            return Projection(oriented)
