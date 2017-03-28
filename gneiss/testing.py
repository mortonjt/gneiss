# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
import numpy as np


def block_diagonal(ncols, nrows, nblocks):
    """ Generate block diagonal with uniformly distributed values within blocks """
    mat = np.zeros((nrows, ncols))
    block_cols = ncols // nblocks
    block_rows = nrows // nblocks
    for b in range(nblocks-1):
        B = np.random.uniform(size=(block_rows, block_cols))
        lower_row = block_rows * b
        upper_row = min(block_rows*(b+1), nrows)
        lower_col = block_cols * b
        upper_col = min(block_cols*(b+1), ncols)

        mat[lower_row : upper_row, lower_col : upper_col] = B

    # Make last block fill in the remainder
    B = np.random.uniform(size=(nrows-upper_row, ncols-upper_col))
    mat[upper_row:, upper_col:] = B
    return mat
