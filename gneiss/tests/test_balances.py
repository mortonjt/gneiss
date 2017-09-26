# ----------------------------------------------------------------------------
# Copyright (c) 2016--, gneiss development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function
import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt
from gneiss.balances import (balance_basis, _count_matrix,
                             _balance_basis, _attach_balances,
                             balanceplot,
                             solve_gaussians,
                             _reorder,
                             round_balance)
from gneiss.layouts import default_layout
from skbio import TreeNode
from skbio.util import get_data_path


class TestPlot(unittest.TestCase):

    def test__attach_balances(self):
        tree = TreeNode.read([u"(a,b);"])
        balances = np.array([10])
        res_tree = _attach_balances(balances, tree)
        self.assertEqual(res_tree.weight, 10)

    def test__attach_balances_level_order(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = np.array([10, -10])
        res_tree = _attach_balances(balances, tree)
        self.assertEqual(res_tree.weight, 10)
        self.assertEqual(res_tree.children[0].weight, -10)

    def test__attach_balances_bad_index(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = np.array([10])
        with self.assertRaises(IndexError):
            _attach_balances(balances, tree)

    def test__attach_balances_series(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = pd.Series([10, -10], index=['r', 'c'])
        res_tree = _attach_balances(balances, tree)
        self.assertEqual(res_tree.weight, 10)

    def test__attach_balances_series_bad(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = pd.Series([10, -10])
        with self.assertRaises(KeyError):
            _attach_balances(balances, tree)

    def test_balanceplot(self):
        tree = TreeNode.read([u"((a,b)c,d)r;"])
        balances = np.array([10, -10])
        tr, ts = balanceplot(balances, tree)
        self.assertEquals(ts.mode, 'c')
        self.assertEquals(ts.layout_fn[0], default_layout)


class TestBalances(unittest.TestCase):

    def test_count_matrix_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])
        res, _ = _count_matrix(t)
        exp = {'k': 0, 'l': 1, 'r': 1, 't': 0, 'tips': 2}
        self.assertEqual(res[t], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[0]], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[1]], exp)

    def test_count_matrix_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        res, _ = _count_matrix(t)

        exp = {'k': 0, 'l': 2, 'r': 1, 't': 0, 'tips': 3}
        self.assertEqual(res[t], exp)
        exp = {'k': 1, 'l': 1, 'r': 1, 't': 0, 'tips': 2}
        self.assertEqual(res[t[0]], exp)

        exp = {'k': 0, 'l': 0, 'r': 0, 't': 0, 'tips': 1}
        self.assertEqual(res[t[1]], exp)
        self.assertEqual(res[t[0][0]], exp)
        self.assertEqual(res[t[0][1]], exp)

    def test_count_matrix_singleton_error(self):
        with self.assertRaises(ValueError):
            tree = u"(((a,b)c, d)root);"
            t = TreeNode.read([tree])
            _count_matrix(t)

    def test_count_matrix_trifurcating_error(self):
        with self.assertRaises(ValueError):
            tree = u"((a,b,e)c, d);"
            t = TreeNode.read([tree])
            _count_matrix(t)

    def test__balance_basis_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])

        exp_basis = np.array([[-np.sqrt(1. / 2), np.sqrt(1. / 2)]])
        exp_keys = [t]
        res_basis, res_keys = _balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test__balance_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])

        exp_basis = np.array([[-np.sqrt(1. / 6), -np.sqrt(1. / 6),
                               np.sqrt(2. / 3)],
                              [-np.sqrt(1. / 2), np.sqrt(1. / 2), 0]
                              ])
        exp_keys = [t, t[0]]
        res_basis, res_keys = _balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_balance_basis_base_case(self):
        tree = u"(a,b);"
        t = TreeNode.read([tree])
        exp_keys = [t]
        exp_basis = np.array([0.19557032, 0.80442968])
        res_basis, res_keys = balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, res_keys)

    def test_balance_basis_unbalanced(self):
        tree = u"((a,b)c, d);"
        t = TreeNode.read([tree])
        exp_keys = [t, t[0]]
        exp_basis = np.array([[0.18507216, 0.18507216, 0.62985567],
                              [0.14002925, 0.57597535, 0.28399541]])

        res_basis, res_keys = balance_basis(t)

        npt.assert_allclose(exp_basis, res_basis)
        self.assertListEqual(exp_keys, list(res_keys))

    def test_balance_basis_large1(self):
        fname = get_data_path('large_tree.nwk',
                              subfolder='data')
        t = TreeNode.read(fname)
        # note that the basis is in reverse level order
        exp_basis = np.loadtxt(
            get_data_path('large_tree_basis.txt',
                          subfolder='data'))
        res_basis, res_keys = balance_basis(t)
        npt.assert_allclose(exp_basis[:, ::-1], res_basis)


class TestGMM(unittest.TestCase):
    def setUp(self):
        self.data = np.array(
            [[0.4579667, 0.39576221, -0.39520894, -0.44957529, 0.0147894,
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
              0.04435679, -0.01826977, -0.01877417, 0.0629691]]).T

    def test_round_balance(self):
        l, r = round_balance(self.data, random_state=0)
        exp_l, exp_r = -0.28480468061713338, 0.14953669163963104
        self.assertAlmostEqual(l, exp_l)
        self.assertAlmostEqual(r, exp_r)


if __name__ == "__main__":
    unittest.main()
