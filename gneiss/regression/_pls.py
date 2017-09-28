# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
from collections import OrderedDict
import numpy as np
import pandas as pd
from gneiss.regression._model import RegressionModel
from gneiss.util import _type_cast_to_float

from gneiss.balances import (solve_gaussians,_reorder,
                             round_balance)

from skbio.stats.composition import clr, centralize

from statsmodels.iolib.summary2 import Summary
from statsmodels.sandbox.tools.cross_val import LeaveOneOut
from patsy import dmatrix
from scipy import stats
from scipy.stats import linregress, spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def pls(table, metadata):
    """ Partial Least Squares applied to balances.

    An partial least squares (PLS) regression is a method for estimating
    parameters in a linear regression model while simultaneously performing
    dimensionality reduction and variable selection.

    Here, we will first perform the clr transform on the original table
    of counts (`table`). Partial Least Squares is then applied to this
    transformed table with  specified covariates (`metadata`).
    The resulting loadings matrix is then fitted using Gaussian Mixture Models.
    These will filter out contaminants, and choose features for the numerator
    and the denominator.

    Cross validation and goodness-of-fit measures are calculated to
    confirm to classification / regression accuracy.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table where samples correspond to rows and
        abundances correspond to columns.  There cannot be any zeros
        in the table.
    metadata: pd.Series
        Metadata table that contains information about the samples contained
        in the `table` object.  Samples correspond to rows and covariates
        correspond to columns.  If categorical variables are specified,
        only two classes can be inputted.

    Returns
    -------
    PLSModel
        Container object that holds information about the overall fit.
        This includes information about coefficients, pvalues, residuals
        and coefficient of determination from the resulting regression.
    """
    ctable, md = table.align(metadata.copy(), join='inner', axis=0)
    if len(md.value_counts()) > 2:
        return PLSRegressor()
    else:
        return PLSClassifier()


def _run_pls(Y, X):
    plsc = PLSRegression(n_components=1)
    if len(X.shape) == 1:
        plsc.fit(X=X.reshape(-1, 1), Y=Y)
    else:
        plsc.fit(X=X, Y=Y)
    return plsc


class PLSClassifier():
    def __init__(self):
        pass

    def fit(self, Y, X, num_folds=4, random_state=None, **kwargs):
        """ Fits PLS classifier

        Parameters
        ----------
        X : np.array
            Independent variable where the rows are samples
            and the columns are features.
        Y : np.array
            Dependent variables, where rows are samples and
            the columns are covariates.

        Returns
        -------
        auroc : np.float
            Area under the curve - also a proxy for accuracy.
        cv : pd.DataFrame
            Cross validation results for the classifier.
        """
        cX = pd.DataFrame(clr(centralize(X)),
                          index=X.index, columns=X.columns)

        Y, cX = Y.align(cX, join='inner', axis=0)
        # build model on entire dataset
        self.plsc = _run_pls(Y, cX)
        pls_df = pd.DataFrame(
            self.plsc.x_weights_, index=X.columns,
            columns=['PLS1'])
        l, r = round_balance(pls_df, random_state=random_state, **kwargs)
        denom = pls_df.loc[pls_df.PLS1 < l]
        num = pls_df.loc[pls_df.PLS1 > r]
        r_, s_ = len(num), len(denom)
        b = (np.log(X.loc[:, num.index]).mean(axis=1) -
             np.log(X.loc[:, denom.index]).mean(axis=1))

        b = b * np.sqrt(r_ * s_ / (r_ + s_))

        group_fpr, group_tpr, thresholds = roc_curve(
            y_true=Y,
            y_score=b)
        auroc = auc(group_tpr, group_fpr)

        # calculate the AUC for the reverse - remember
        # a perfectly bad classifer can be turned into a
        # perfected good classifier.
        flipped_fpr, flipped_tpr, thresholds = roc_curve(
            y_true=1-Y,
            y_score=b)
        auroc_flipped = auc(flipped_tpr, flipped_fpr)

        self.balance = b

        if auroc < auroc_flipped:
            group_fpr, group_tpr = flipped_fpr, flipped_tpr
            Y = 1 - Y
            auroc = auroc_flipped

        f = lambda x, y: euclidean(x, (1 - y))
        E = np.vstack((group_fpr, 1 - group_tpr))
        E = np.sum(E**2, axis=0)

        i = np.argmin(E)
        self.threshold = thresholds[i]
        self.numerator = list(num.index)
        self.denominator = list(denom.index)
        return auroc, self.kfold(metadata=Y, table=X, num_folds=num_folds,
                                 random_state=random_state, **kwargs)

    def predict(self, X=None, **kwargs):
        """ Perform prediction based on PLS model. """
        return self.plsc.predict(X, **kwargs)

    def summary(self):
        pass

    def kfold(self, table, metadata, num_folds, random_state=None, **kwargs):
        """ Performs cross validation, returning AUROC statistic. """
        table, metadata = table.align(metadata, join='inner', axis=0)
        skf = KFold(n_splits=num_folds, shuffle=True,
                    random_state=random_state)
        ctable = pd.DataFrame(clr(centralize(table)),
                              index=table.index, columns=table.columns)

        cv = pd.DataFrame(columns=['AUROC'], index=np.arange(num_folds),
                          dtype=np.float)
        for i, (train, test) in enumerate(skf.split(ctable.values,
                                                    metadata.values)):

            X_train, X_test = ctable.iloc[train], ctable.iloc[test]
            Y_train, Y_test = metadata.iloc[train], metadata.iloc[test]
            plsc = _run_pls(Y_train, X_train)
            pls_df = pd.DataFrame(plsc.x_weights_, index=ctable.columns,
                                  columns=['PLS1'])

            l, r = round_balance(pls_df, random_state=random_state, **kwargs)
            denom = pls_df.loc[pls_df.PLS1 < l]
            num = pls_df.loc[pls_df.PLS1 > r]

            # make the prediction and evaluate the accuracy
            idx = table.index[test]
            b = (np.log(table.loc[idx, num.index]).mean(axis=1) -
                 np.log(table.loc[idx, denom.index]).mean(axis=1))
            r_, s_ = len(num), len(denom)
            b = b * np.sqrt(r_ * s_ / (r_ + s_))

            group_fpr, group_tpr, thresholds = roc_curve(
                y_true=Y_test,
                y_score=b)

            auroc = auc(group_tpr, group_fpr)
            cv.loc[i, 'AUROC'] = auroc.astype(np.float)

        return cv


class PLSRegressor(RegressionModel):
    def __init__(self):
        pass

    def fit(self, Y, X, num_folds=4, **kwargs):
        """ Fits PLS Regressor

        Parameters
        ----------
        X : np.array
            Independent variable where the rows are samples
            and the columns are features.
        Y : np.array
            Dependent variables, where rows are samples and
            the columns are covariates.

        """
        Y, X = Y.align(X, join='inner', axis=0)
        cX = pd.DataFrame(clr(centralize(X)),
                          index=X.index, columns=X.columns)

        # build model on entire dataset
        self.plsc = _run_pls(Y, cX)

        pls_df = pd.DataFrame(
            self.plsc.x_weights_, index=X.columns,
            columns=['PLS1'])
        l, r = round_balance(pls_df, **kwargs)
        denom = pls_df.loc[pls_df.PLS1 < l]
        num = pls_df.loc[pls_df.PLS1 > r]

        r_, s_ = len(num), len(denom)
        b = (np.log(X.loc[:, num.index]).mean(axis=1) -
             np.log(X.loc[:, denom.index]).mean(axis=1))
        self.balance = b * np.sqrt(r_ * s_ / (r_ + s_))
        r2 = spearmanr(b, Y)

        self.numerator = list(num.index)
        self.denominator = list(denom.index)
        return r2, self.kfold(metadata=Y, table=X,
                              num_folds=num_folds, **kwargs)

    def predict(self, X=None, **kwargs):
        """ Perform prediction based on PLS model. """
        return self.plsc.predict(X, **kwargs)

    def summary(self):
        pass

    def kfold(self, table, metadata, num_folds, random_state=None,
              **kwargs):
        """ Performs cross validation, returning Q2 statistic. """
        skf = KFold(n_splits=num_folds, shuffle=True,
                    random_state=random_state)
        ctable = pd.DataFrame(clr(centralize(table)),
                              index=table.index, columns=table.columns)

        cv = pd.DataFrame(columns=['Q2'], index=np.arange(num_folds),
                          dtype=np.float)
        for i, (train, test) in enumerate(skf.split(ctable.values,
                                                    metadata.values)):

            X_train, X_test = ctable.iloc[train], ctable.iloc[test]
            Y_train, Y_test = metadata.iloc[train], metadata.iloc[test]

            plsc = _run_pls(Y_train, X_train)

            pls_df = pd.DataFrame(plsc.x_weights_, index=ctable.columns,
                                  columns=['PLS1'])

            l, r = round_balance(pls_df, random_state=random_state, **kwargs)
            denom = pls_df.loc[pls_df.PLS1 < l]
            num = pls_df.loc[pls_df.PLS1 > r]

            # make the prediction and evaluate the accuracy
            idx = table.index[train]
            b = (np.log(table.loc[idx, num.index]).mean(axis=1) -
                 np.log(table.loc[idx, denom.index]).mean(axis=1))
            r_, s_ = len(num), len(denom)
            balance = b * np.sqrt(r_ * s_ / (r_ + s_))

            b_, int_, _, _, _ = linregress(balance, Y_train)

            idx = table.index[test]
            balance = (np.log(table.loc[idx, num.index] + 1).mean(axis=1) -
                       np.log(table.loc[idx, denom.index] + 1).mean(axis=1))
            pred = balance * b_ + int_

            press = ((pred - Y_test)**2).sum()
            tss = ((Y_test.mean() - Y_test)**2).sum()
            Q2 = 1 - (press / tss)
            cv.loc[i, 'Q2'] = Q2


        return cv

