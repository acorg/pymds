#!/usr/bin/env python3

import unittest
import numpy as np
from scipy.spatial.distance import pdist, squareform

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

if __name__ == '__main__':
    unittest.main()
