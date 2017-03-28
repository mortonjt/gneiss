# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
from gneiss.testing import block_diagonal
import numpy.testing as npt
import numpy as np


class TestTestingUtils(unittest.TestCase):

    def test_block_diagonal_4x4(self):
        np.random.seed(0)
        res = block_diagonal(4, 4, 2)
        exp = np.array([[0.5488135, 0.71518937, 0., 0.],
                        [0.60276338, 0.54488318, 0., 0.],
                        [0., 0., 0.4236548, 0.64589411],
                        [0., 0., 0.43758721, 0.891773]])
        npt.assert_allclose(res, exp, rtol=1e-5, atol=1e-5)

    def test_block_diagonal_3x4(self):
        np.random.seed(0)
        res = block_diagonal(3, 4, 2)
        exp = np.array([[0.5488135, 0.71518937, 0., 0.],
                        [0., 0., 0.60276338, 0.54488318],
                        [0., 0., 0.4236548, 0.64589411]])
        npt.assert_allclose(res, exp, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
