#!/usr/bin/env python3

import os
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import pymds
from pymds.mds import DistanceMatrix


class TestDistanceMatrixInit(unittest.TestCase):
    """Tests for pymds.mds.DistanceMatrix."""

    def setUp(self):
        np.random.seed(1234)
        dist = squareform(pdist(np.random.randn(3, 2)))
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
        dist = pd.DataFrame(squareform(pdist(np.random.randn(3, 2))))
        dm = DistanceMatrix(dist)
        self.assertIsInstance(dm.D, np.ndarray)

    def test_D_ndarray_when_passed_ndarray(self):
        """DistanceMatrix.D should always be a np.array."""
        dist = squareform(pdist(np.random.randn(3, 2)))
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
        dist = squareform(pdist(np.random.randn(3, 2)))
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


if __name__ == '__main__':
    unittest.main()
