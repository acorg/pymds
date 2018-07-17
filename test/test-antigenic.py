#!/usr/bin/env python3

import os
import unittest

import numpy as np
import pandas as pd

import pymds
from pymds.antigenic import Table


class TestTable(unittest.TestCase):
    """Tests for pymds.antigenic.Table."""

    def test_D_has_correct_shape(self):
        df = pd.DataFrame(
            2 ** np.random.randint(low=0, high=8, size=(10, 4)) * 10)
        t = Table(df)
        self.assertEqual((14, 14), t.D.shape)

    def test_optimize(self):
        path = os.path.join(
            pymds.__path__[0], '..', 'test', 'data', 'hi_table.csv')
        df = pd.read_csv(path, index_col=0)
        self.assertTrue(False)  # Have to handle <10 values

if __name__ == '__main__':
    unittest.main()
