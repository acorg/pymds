#!/usr/bin/env python3

import os
import unittest

import numpy as np
from numpy.random import randn
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform

import pymds
from pymds.mds import DistanceMatrix, Projection


class TestDistanceMatrixInit(unittest.TestCase):
    """Tests for pymds.mds.DistanceMatrix."""

    def setUp(self):
        np.random.seed(1234)
        dist = squareform(pdist(randn(3, 2)))
        self.dm = DistanceMatrix(dist)

    def test_has_D(self):
        self.assertTrue(hasattr(self.dm, 'D'))

    def test_1d(self):
        with self.assertRaises(ValueError):
            DistanceMatrix(np.ones(2))

    def test_not_square(self):
        with self.assertRaises(ValueError):
            DistanceMatrix(np.ones((1, 2)))

    def test_m_is_number_of_samples(self):
        self.assertEqual(3, self.dm.m)

    def test_D_ndarray_when_passed_df(self):
        """DistanceMatrix.D should always be a np.array."""
        dist = DataFrame(squareform(pdist(randn(3, 2))))
        dm = DistanceMatrix(dist)
        self.assertIsInstance(dm.D, np.ndarray)

    def test_D_ndarray_when_passed_ndarray(self):
        """DistanceMatrix.D should always be a np.array."""
        dist = squareform(pdist(randn(3, 2)))
        dm = DistanceMatrix(dist)
        self.assertIsInstance(dm.D, np.ndarray)

    def test_D_ndarray_when_passed_str(self):
        """DistanceMatrix.D should always be a np.array."""
        path = os.path.join(
            pymds.__path__[0],
            '..',
            'test',
            'data',
            '10x10-distance-matrix.csv')
        dm = DistanceMatrix(path)
        self.assertIsInstance(dm.D, np.ndarray)


class TestDistanceMatrixError(unittest.TestCase):
    """Tests for pymds.mds.DistanceMatrix._error"""

    def setUp(self):
        np.random.seed(1234)
        dist = squareform(pdist(randn(3, 2)))
        self.dm = DistanceMatrix(dist)

    def test_zero_diff(self):
        """Error should be zero when all elements in diff are 0"""
        diff = np.zeros_like(self.dm.D)
        self.assertEqual(0, self.dm._error(diff))

    def test_two_diff(self):
        """Error should be 0.5 * 4 * m ** 2 when all elements in diff are 2"""
        diff = np.ones_like(self.dm.D) * 2
        m = self.dm.D.shape[0]
        self.assertEqual(0.5 * 4 * m ** 2, self.dm._error(diff))


class TestDistanceMatrixEmbed(unittest.TestCase):
    """Tests for pymds.mds.DistanceMatrix.embed"""

    def test_triangle(self):
        """Three samples, each at 1 unit from each other. Expect 0 stress."""
        dist = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        dm = DistanceMatrix(dist)
        projection = dm.embed()
        self.assertAlmostEqual(0, projection.stress)


class TestProjectionSharedWith(unittest.TestCase):
    """Tests for pymds.mds.Projection._get_samples_shared_with()"""

    def test_raises_error_if_no_shared_idx(self):
        a = Projection(DataFrame(randn(10, 2), index=range(0, 10)))
        b = Projection(DataFrame(randn(10, 2), index=range(10, 20)))
        with self.assertRaises(ValueError):
            a._get_samples_shared_with(b)

    def test_raises_error_if_dupes_in_coord_index(self):
        index = list(range(9)) + [4]
        a = Projection(DataFrame(randn(10, 2), index=index))
        b = Projection(DataFrame(randn(10, 2)))
        with self.assertRaises(ValueError):
            a._get_samples_shared_with(b)

    def test_raises_error_if_dupes_in_other_index(self):
        index = list(range(9)) + [4]
        a = Projection(DataFrame(randn(10, 2)))
        b = Projection(DataFrame(randn(10, 2), index=index))
        with self.assertRaises(ValueError):
            a._get_samples_shared_with(b)

    def test_raises_error_if_dupes_in_index_arg(self):
        a = Projection(DataFrame(randn(10, 2)))
        b = Projection(DataFrame(randn(10, 2)))
        with self.assertRaises(ValueError):
            a._get_samples_shared_with(b, index=[1, 1])

    def test_raises_error_if_elements_in_index_not_in_projection(self):
        a = Projection(DataFrame(randn(10, 2)))
        b = Projection(DataFrame(randn(10, 2), index=range(5, 15)))
        with self.assertRaises(ValueError):
            a._get_samples_shared_with(b, index=[14])

    def test_raises_error_if_elements_in_index_not_in_other(self):
        a = Projection(DataFrame(randn(10, 2)))
        b = Projection(DataFrame(randn(10, 2), index=range(5, 15)))
        with self.assertRaises(ValueError):
            a._get_samples_shared_with(b, index=[4])

    def test_raises_error_if_other_is_array_wrong_shape(self):
        a = Projection(DataFrame(randn(10, 2)))
        with self.assertRaises(ValueError):
            a._get_samples_shared_with(randn(11, 2))


class TestProjectionOrientTo(unittest.TestCase):
    """Tests for pymds.mds.Projection.orient_to"""

    def test_returns_projection(self):
        a = Projection(DataFrame(randn(10, 2)))
        b = Projection(DataFrame(randn(10, 2)))
        self.assertIsInstance(a.orient_to(b), Projection)

    def test_orients_correct_if_other_is_arr(self):
        arr = randn(10, 2)
        # Flip arr left-right, rotate 90 deg counter clockwise, and translate
        other = np.fliplr(arr)
        other = np.dot(other, np.array([[0, -1], [1, 0]]))
        other += np.array([10, -5])

        oriented = Projection(DataFrame(arr)).orient_to(other)
        self.assertAlmostEqual(0, (oriented.coords.values - other).sum())

    def test_orients_correct_if_other_is_projection(self):
        arr = randn(10, 2)
        # Flip arr left-right, rotate 90 deg counter clockwise, and translate
        other = np.fliplr(arr)
        other = np.dot(other, np.array([[0, -1], [1, 0]]))
        other += np.array([10, -5])
        other = Projection(DataFrame(other))

        oriented = Projection(DataFrame(arr)).orient_to(other)
        diff = (oriented.coords.values - other.coords.values).sum()
        self.assertAlmostEqual(0, diff)

    def test_orients_on_subset_returns_full_data(self):
        a = Projection(DataFrame(randn(10, 2)))
        b = Projection(DataFrame(randn(10, 2)))
        oriented = a.orient_to(b, index=range(4))
        self.assertEqual(10, len(oriented.coords.index))


if __name__ == '__main__':
    unittest.main()
