# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from gneiss.classification._model import ClassificationModel
from patsy import dmatrix
from statsmodels.discrete.discrete_model import MNLogit
from gneiss.balances import balance_basis
from gneiss.util import rename_internal_nodes


def mnlogit(outcome, balances, tree, **kwargs):
    """ Multinomial logistic regression on balances.

    A multinomial logistic regression is applied to balances.
    This technique can be utilized as a classification technique.
    The relative abundance data is transformed into balances using the
    ILR transformation, using a tree to specify the groupings of the features.
    The regression is then performed on each balance separately. Only positive
    data will be accepted, so if there are zeros present, consider using a zero
    imputation method such as ``multiplicative_replacement`` or add a
    pseudocount.

    Parameters
    ----------
    outcome : pd.Series
        The outcome to be predicted by balances.
    balances : pd.DataFrame
        Contingency table where samples correspond to rows and
        features correspond to columns.  The features correspond to balances.
    tree : skbio.TreeNode
        Tree object that defines the partitions of the features. Each of the
        leaves correspond to the columns contained in the table.

    See Also
    --------
    statsmodels.discrete.discrete_model.MNLogit
    """
    # TODO: will need to make sure that the balance names
    # match up with the internal node names in the tree.

    # TODO: raise warning if they don't match up exactly.
    balances, outcome = balances.align(outcome, join='inner', axis=0)
    model = MNLogit(outcome, balances)

    basis, _ = balance_basis(tree)

    return MNLogitModel(basis=basis, tree=tree, balances=balances, submodels=model)


class MNLogitModel(ClassificationModel):
    """ Summary object for storing multinomial logistic regression results.

    A `MNLogitModel` stores information about the individual balances
    used in the classification, the class probabilities, coefficients and
    along with test statistics for each of the coefficients. This object can
    also be used to perform predictions.

    Attributes
    ----------
    submodels : statsmodels.discrete.discrete_model.MNLogit
        The multinomial logistic regression summary object
    basis : pd.DataFrame
        Orthonormal basis in the Aitchison simplex.
        Row names correspond to the leaves of the tree
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, regularized=False, **kwargs):
        """ Fit the model.

        Parameters
        ----------
        regularized : bool
            Specifies if a regularization procedure should be used
            when performing the fit. (default = False)
        **kwargs : dict
           Keyword arguments used to tune the parameter estimation.
        """
        # there is only one big model here.
        if not regularized:
            self.results = self.submodels.fit(**kwargs)
        else:
            self.results = self.submodels.fit_regularized(**kwargs)

    def roc(self):
        pass

    def kfold(self):
        pass

    def summary(self):
        pass

    def predict(self):
        pass
