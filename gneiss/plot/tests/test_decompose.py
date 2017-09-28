# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------
import unittest
from gneiss.plot import (balance_boxplot, balance_barplots,
                         balance_histogram, mixture_plot, 
                         proportion_plot)
import numpy as np
import pandas as pd
import numpy.testing as npt
import matplotlib.pyplot as plt
from skbio import TreeNode
from skbio.util import get_data_path


class TestHistogram(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'y': [-2, -2.2, -1.8, -1.5, -1, 1, 1.5, 2, 2.2, 1.8],
            'group': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']
            })
    def test_histogram(self):
        a = balance_histogram(self.df.y, self.df.group)
        res = a.get_lines()[0]._xy
        exp = np.loadtxt(get_data_path('test_histogram_coords.txt'))
        npt.assert_allclose(exp, res)


class TestBoxplot(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'y': [-2, -2.2, -1.8, -1.5, -1, 1, 1.5, 2, 2.2, 1.8],
            'group': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
            'hue': ['0', '1', '0', '1', '0', '1', '0', '1', '0', '1']}
        )
        self.tree = TreeNode.read(['((c, d)z, (b,a)x)y;'])
        self.feature_df = pd.DataFrame(
            {
                'type': ['tomato', 'carrots', 'apple', 'bacon'],
                'food': ['vegatable', 'vegatable', 'fruit', 'meat'],
                'seed': ['yes', 'no', 'yes', 'no']
            },
            index=["a", "b", "c", "d"]
        )

    def test_basic_boxplot(self):
        a = balance_boxplot('y', y='group', data=self.df)
        res = np.vstack([i._xy for i in a.get_lines()])
        exp = np.array([[-2., 0.],
                        [-2.2, 0.],
                        [-1.5, 0.],
                        [-1., 0.],
                        [-2.2, -0.2],
                        [-2.2, 0.2],
                        [-1., -0.2],
                        [-1., 0.2],
                        [-1.8, -0.4],
                        [-1.8, 0.4],
                        [1.5, 1.],
                        [1., 1.],
                        [2., 1.],
                        [2.2, 1.],
                        [1., 0.8],
                        [1., 1.2],
                        [2.2, 0.8],
                        [2.2, 1.2],
                        [1.8, 0.6],
                        [1.8, 1.4]])
        npt.assert_allclose(res, exp)

    def test_basic_hue_boxplot(self):
        a = balance_boxplot('y', y='group', hue='hue', data=self.df)
        res = np.vstack([i._xy for i in a.get_lines()])
        exp = np.array([[-1.9, -0.2],
                        [-2., -0.2],
                        [-1.4, -0.2],
                        [-1., -0.2],
                        [-2., -0.298],
                        [-2., -0.102],
                        [-1., -0.298],
                        [-1., -0.102],
                        [-1.8, -0.396],
                        [-1.8, -0.004],
                        [-2.025, 0.2],
                        [-2.2, 0.2],
                        [-1.675, 0.2],
                        [-1.5, 0.2],
                        [-2.2, 0.102],
                        [-2.2, 0.298],
                        [-1.5, 0.102],
                        [-1.5, 0.298],
                        [-1.85, 0.004],
                        [-1.85, 0.396],
                        [1.675, 0.8],
                        [1.5, 0.8],
                        [2.025, 0.8],
                        [2.2, 0.8],
                        [1.5, 0.702],
                        [1.5, 0.898],
                        [2.2, 0.702],
                        [2.2, 0.898],
                        [1.85, 0.604],
                        [1.85, 0.996],
                        [1.4, 1.2],
                        [1., 1.2],
                        [1.9, 1.2],
                        [2., 1.2],
                        [1., 1.102],
                        [1., 1.298],
                        [2., 1.102],
                        [2., 1.298],
                        [1.8, 1.004],
                        [1.8, 1.396]])
        npt.assert_allclose(res, exp)

    def test_basic_barplot(self):
        ax_denom, ax_num = balance_barplots(self.tree, 'y', header='food',
                                            feature_metadata=self.feature_df)


class TestProportionPlot(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({
            'A': [1, 1.2, 1.1, 2.1, 2.2, 2],
            'B': [9.9, 10, 10.1, 2, 2.4, 2.1],
            'C': [5, 3, 1, 2, 2, 3],
            'D': [5, 5, 5, 5, 5, 5],
        }, index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

        self.feature_metadata = pd.DataFrame({
            'A': ['k__foo', 'p__bar', 'c__', 'o__', 'f__', 'g__', 's__'],
            'B': ['k__foo', 'p__bar', 'c__', 'o__', 'f__', 'g__', 's__'],
            'C': ['k__poo', 'p__tar', 'c__', 'o__', 'f__', 'g__', 's__'],
            'D': ['k__poo', 'p__far', 'c__', 'o__', 'f__', 'g__', 's__']
        }, index=['kingdom', 'phylum', 'class', 'order',
                  'family', 'genus', 'species']).T

        self.metadata = pd.DataFrame({
            'groups': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'dry': [1, 2, 3, 4, 5, 6]
        }, index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])

    def test_proportion_plot(self):
        np.random.seed(0)
        num_features = ['A', 'B']
        denom_features = ['C', 'D']
        ax1, ax2 = proportion_plot(self.table, self.metadata,
                                   'groups', 'X', 'Y',
                                   num_features, denom_features,
                                   self.feature_metadata,
                                   label_col='phylum')
        res = np.vstack([l.get_xydata() for l in ax1.get_lines()])
        exp=np.array([0., 0., 1., 1., 2., 2., 3., 3.])

        npt.assert_allclose(res[:, 1], exp, verbose=True)

        res = np.vstack([l.get_xydata() for l in ax2.get_lines()])
        exp=np.array([0., 0., 1., 1., 2., 2., 3., 3.])

        npt.assert_allclose(res[:, 1], exp, verbose=True)

        res = [l._text for l in ax2.get_yticklabels()]
        exp = ['p__bar', 'p__bar', 'p__tar', 'p__far']
        self.assertListEqual(res, exp)

    def test_proportion_plot_order(self):
        self.maxDiff = None
        np.random.seed(0)
        # tests for different ordering
        num_features = ['A', 'B']
        denom_features = ['D', 'C']
        ax1, ax2 = proportion_plot(self.table, self.metadata,
                                   'groups', 'X', 'Y',
                                   num_features, denom_features,
                                   self.feature_metadata,
                                   label_col='phylum')
        res = np.vstack([l.get_xydata() for l in ax1.get_lines()])
        exp = np.array([0., 0., 1., 1., 2., 2., 3., 3.])

        npt.assert_allclose(res[:, 1], exp, atol=1e-2, rtol=1e-2, verbose=True)

        res = np.vstack([l.get_xydata() for l in ax2.get_lines()])
        exp = np.array([0., 0., 1., 1., 2., 2., 3., 3.])

        npt.assert_allclose(res[:, 1], exp, atol=1e-2, rtol=1e-2, verbose=True)

        res = [l._text for l in ax2.get_yticklabels()]
        exp = ['p__bar', 'p__bar', 'p__far', 'p__tar']
        self.assertListEqual(res, exp)

    def test_proportion_plot_order_figure(self):
        self.maxDiff = None
        np.random.seed(0)
        # tests for different ordering
        fig, axes = plt.subplots(1, 2)

        num_features = ['A', 'B']
        denom_features = ['D', 'C']
        ax1, ax2 = proportion_plot(self.table, self.metadata,
                                   'groups', 'X', 'Y',
                                   num_features, denom_features,
                                   self.feature_metadata,
                                   label_col='phylum', axes=axes)
        res = np.vstack([l.get_xydata() for l in ax1.get_lines()])
        exp = np.array([0., 0., 1., 1., 2., 2., 3., 3.])

        npt.assert_allclose(res[:, 1], exp, atol=1e-2, rtol=1e-2, verbose=True)

        res = np.vstack([l.get_xydata() for l in ax2.get_lines()])
        exp = np.array([0., 0., 1., 1., 2., 2., 3., 3.])


        npt.assert_allclose(res[:, 1], exp, atol=1e-2, rtol=1e-2, verbose=True)

        res = [l._text for l in ax2.get_yticklabels()]
        exp = ['p__bar', 'p__bar', 'p__far', 'p__tar']
        self.assertListEqual(res, exp)

    def test_proportion_plot_original_labels(self):
        # tests for different ordering
        fig, axes = plt.subplots(1, 2)

        num_features = ['A', 'B']
        denom_features = ['D', 'C']
        ax1, ax2 = proportion_plot(self.table, self.metadata,
                                   'groups', 'X', 'Y',
                                   num_features, denom_features,
                                   axes=axes)

        res = [l._text for l in ax2.get_yticklabels()]
        exp = ['A', 'B', 'D', 'C']
        self.assertListEqual(res, exp)


class TestMixturePlot(unittest.TestCase):

    def setUp(self):
        self.spectrum = np.array(
            [0.4579667, 0.39576221, -0.39520894, -0.44957529, 0.0147894,
             0.03894491, 0.01431574, 0.02130079, 0.00530691, 0.00163606,
             0.10417575, 0.01679624, 0.01408287, 0.08377197, -0.05097175,
             0.01467061, 0.0513028, 0.03894267, 0.07682788, -0.02302689,
             0.03727277, -0.00167041, 0.06700641, 0.09992187, 0.07400123,
             -0.05075235, 0.03855951, 0.03232991, 0.033296, -0.0778636,
             -0.02262944, 0.01665713, 0.02012388, -0.08734141, 0.04402584,
             0.01885096, 0.01236461, 0.02019468, -0.01489146, -0.10339335,
             -0.0526063, -0.03070242, 0.01214559, -0.15510279, -0.04290816,
             0.04884383, 0.03615357, -0.00967101, 0.02681241, 0.01047964,
             -0.03984972, -0.0016186, 0.02497351, -0.02950191, 0.04832895,
             -0.068324, 0.00458738, 0.01106375, 0.04545569, 0.00771012,
             0.02453104, -0.01616486, 0.05563585, 0.01309359, 0.01579368,
             0.0051668, 0.01042911, -0.07541249, -0.0228381, -0.00250977,
             -0.0163356, -0.11578245, 0.00780789, -0.04505144, 0.11493317,
             0.06772574, -0.06261561, -0.08941559, -0.02147429, -0.01220844,
             -0.04686819, 0.05811476, -0.02413633, 0.14336764, -0.08111341,
             0.05834844, -0.09425382, 0.03425244, 0.05037963, -0.0336687,
             -0.06270773, 0.07621378, 0.04144562, 0.01764233, 0.05221101,
             -0.04337608, 0.06173909, -0.04485265, 0.01397837, 0.04435679,
             0.04435679, -0.01826977, -0.01877417, 0.0629691 ])

    def test_mixture_plot(self):
        a = mixture_plot(self.spectrum)
        res = a.get_lines()[0]._xy
        exp = np.loadtxt(get_data_path('test_mixture_coords.txt'))
        npt.assert_allclose(exp, res)


if __name__ == '__main__':
    unittest.main()
