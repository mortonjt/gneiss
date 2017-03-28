# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import abc
import pandas as pd
from skbio.stats.composition import ilr_inv
from gneiss._model import Model


class ClassificationModel(Model):
    def __init__(self, *args, **kwargs):
        """
        Summary object for storing classification results

        A `RegressionModel` object stores information about the
        individual balances used in the classification, the parameters,
        other summary statistics. This object can be used to perform
        predictions.

        Parameters
        ----------
        submodels : list of statsmodels objects
            List of statsmodels result objects.
        basis : pd.DataFrame
            Orthonormal basis in the Aitchison simplex.
            Row names correspond to the leafs of the tree
            and the column names correspond to the internal nodes
            in the tree. If this is not specified, then `project` cannot
            be enabled in `coefficients` or `predict`.
        tree : skbio.TreeNode
            Bifurcating tree that defines `basis`.
        balances : pd.DataFrame
            A table of balances where samples are rows and
            balances are columns.  These balances were calculated
            using `tree`.
        """
        super().__init__(*args, **kwargs)


    @abc.abstractmethod
    def predict(self, **kwargs):
        pass

    @abc.abstractmethod
    def kfold(self, nfolds=5, **kwargs):
        pass

    @abc.abstractmethod
    def roc(self):
        pass
