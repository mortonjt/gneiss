# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
from skbio.stats.composition import ilr_inv
from skbio import TreeNode
from skbio.util import get_data_path
from gneiss.classification._mnlogit import mnlogit
from gneiss.testing import block_diagonal
from gneiss.cluster import proportional_linkage
from gneiss.stats._stats import ilr_transform
from gneiss.util import rename_internal_nodes


class TestMNLogit(unittest.TestCase):

    def test_mnlogit(self):
        N = 100
        F = 10
        np.random.seed(0)
        mat = block_diagonal(nrows=N, ncols=F, nblocks=2) + 0.0001
        mat = pd.DataFrame(mat, index=np.arange(N).astype(np.str),
                           columns=np.arange(F).astype(np.str))

        outcome = pd.Series([0] * (N//2) + [1] * (N//2),
                            index=np.arange(N).astype(np.str))
        tree = proportional_linkage(mat)
        tree = rename_internal_nodes(tree)
        balances = ilr_transform(table=mat, tree=tree)
        model = mnlogit(outcome, balances, tree)
        model.fit(regularized=True, alpha=0.0001)
        exp = pd.Series({'y0': 1.283722,
                         'y1': 0.000000,
                         'y2': 0.000000,
                         'y3': 0.000000,
                         'y4': 0.000000,
                         'y5': 0.000000,
                         'y6': 0.000000,
                         'y7': 0.000000,
                         'y8': 0.000000})
        npt.assert_allclose(exp.values, np.ravel(model.results.params),
                            atol=0.1, rtol=0.1)


if __name__ == "__main__":
    unittest.main()


