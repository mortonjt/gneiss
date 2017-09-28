# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import os
import shutil
import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
from skbio.stats.composition import ilr_inv
from skbio import TreeNode
from skbio.util import get_data_path
from gneiss.regression import PLSClassifier, PLSRegressor
from gneiss.balances import balance_basis


class TestPLSClassifier(unittest.TestCase):
    def test_pls_simple(self):
        table = pd.read_csv(get_data_path('test_pls_table.csv'),
                            index_col=0)
        metadata = pd.read_csv(get_data_path('test_pls_metadata.csv'),
                               index_col=0)
        model = PLSClassifier()
        auc, cv = model.fit(X=table+1, Y=metadata.Group, num_folds=4,
                            random_state=10)
        exp_numerator = ['0', '1']
        exp_denominator = ['2', '3']
        self.assertListEqual(model.numerator,
                             exp_numerator)
        self.assertListEqual(model.denominator,
                             exp_denominator)

        exp_cv = pd.DataFrame({'AUROC': [1., 1., 1., 1.]},
                              index=[0, 1, 2, 3])

        npt.assert_allclose(exp_cv.AUROC.values, cv.AUROC.values)
        self.assertAlmostEqual(-0.80909719, model.threshold)


class TestPLSRegressor(unittest.TestCase):
    def test_pls_simple(self):
        table = pd.read_csv(get_data_path('test_pls_table.csv'),
                            index_col=0)
        metadata = pd.read_csv(get_data_path('test_pls_metadata.csv'),
                               index_col=0)
        model = PLSRegressor()
        q2, cv = model.fit(X=table+1, Y=metadata.Gradient,
                           num_folds=4, random_state=0)
        exp_numerator = ['2', '3']
        exp_denominator = ['0', '1']
        self.assertListEqual(model.numerator,
                             exp_numerator)
        self.assertListEqual(model.denominator,
                             exp_denominator)

        exp_cv = pd.DataFrame({
            'Q2' : [0.792299525854, 0.769248550073,
                    0.785160649301, 0.796454294479]},
            index=[0, 1, 2, 3])
        pdt.assert_frame_equal(cv, exp_cv)


if __name__=="__main__":
    unittest.main()
