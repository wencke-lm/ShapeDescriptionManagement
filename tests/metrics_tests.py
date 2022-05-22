"""metrics_test.py - verify measures of classification difficulty"""

import os
import sys
import unittest

import numpy as np
import scipy as sp

# in order to access module from sister directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from lib.exceptions import ScarceDataError
from lib.metrics import (
    geometric_separability_index,
    imbalance_ratio,
    minimum_hellinger_distance,
    multiclass_imbalance_ratio,
    number_of_instances,
    self_bleu,
    subgraph_density,
    subset_representativity,
    _brevity_penalty,
    _closest_ref_len,
    _modified_precision,
    _self_bleu
)

class SelfBleuTestCase(unittest.TestCase):
    def test_no_instances(self):
        labels = np.array([
        ])
        uni_count_mtrx = sp.sparse.csr_matrix([
        ])
        with self.assertRaises(ScarceDataError):
            self_bleu([uni_count_mtrx], labels)

    def test_single_instance_for_one_class(self):
        labels = np.array([
            [True, False],
            [False, True],
            [False, True],
            [False, True]
        ])
        # unigram counts
        row = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        col = np.array([0, 1, 2, 4, 0, 2, 3, 4, 1, 2, 3])
        data = np.array([1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 2])
        uni_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(4, 5)
        )
        # [[1, 1, 0, 0, 0]
        #  [0, 0, 3, 0, 1]
        #  [1, 0, 2, 1, 1]
        #  [0, 1, 1, 2, 0]]
        result = self_bleu([uni_count_mtrx], labels)
        expected = [None, 0.75]
        self.assertEqual(result, expected)

    def test_self_bleu_unigrams_bigrams(self):
        labels = np.array([
            [True, False],
            [False, True],
            [True, False],
            [False, True]
        ])
        # unigram counts
        row = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
        col = np.array([1, 4, 5, 2, 3, 4, 5, 0, 1, 0, 3])
        data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1])
        uni_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(4, 6)
        )
        # [[0 1 0 0 1 1]
        #  [0 0 1 1 1 1]
        #  [1 1 0 0 0 0]
        #  [2 0 0 1 0 0]]

        # bigram counts
        row = np.array([0, 0, 1, 1, 1, 2, 3, 3])
        col = np.array([3, 6, 4, 5, 7, 1, 0, 2])
        data = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        bi_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(4, 8)
        )
        # [[0 0 0 1 0 0 1 0]
        #  [0 0 0 0 1 1 0 1]
        #  [0 1 0 0 0 0 0 0]
        #  [1 0 1 0 0 0 0 0]]

        result = self_bleu([uni_count_mtrx, bi_count_mtrx], labels)
        expected = [0.47809144, 0.35634832]
        self.assertEqual(result, expected)

    def test_self_bleu_unigrams(self):
        labels = np.array([
            [True, False],
            [True, False],
            [False, True],
            [False, True]
        ])
        # unigram counts
        row = np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        col = np.array([0, 1, 2, 4, 0, 2, 3, 4, 1, 2, 3])
        data = np.array([1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 2])
        uni_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(4, 5)
        )
        # [[1, 1, 0, 0, 0]
        #  [0, 0, 3, 0, 1]
        #  [1, 0, 2, 1, 1]
        #  [0, 1, 1, 2, 0]]

        result = self_bleu([uni_count_mtrx], labels)
        expected = [0.25, 0.54545455]
        self.assertEqual(result, expected)

    def test_self_bleu_single_intra_class_unigrams_bigrams(self):
        # unigram counts
        row = np.array([0, 0, 1, 1, 2, 2, 2])
        col = np.array([0, 1, 2, 4, 1, 2, 3])
        data = np.array([1, 1, 3, 1, 1, 1, 2])
        uni_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(3, 5)
        )
        # [[1, 1, 0, 0, 0]
        #  [0, 0, 3, 0, 1]
        #  [0, 1, 1, 2, 0]]

        # bigram counts
        row = np.array([0, 1, 1, 2, 2, 2])
        col = np.array([0, 1, 2, 2, 3, 4])
        data = np.array([1, 2, 1, 1, 1, 1])
        bi_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(3, 5)
        )
        # [[1, 0, 0, 0, 0]
        #  [0, 2, 1, 0, 0]
        #  [0, 0, 1, 1, 1]]

        result = _self_bleu([uni_count_mtrx, bi_count_mtrx])
        expected = 0.42481852
        self.assertAlmostEqual(result, expected)

    def test_closest_ref_len(self):
        row = np.array([0, 0, 0, 1, 1, 2])
        col = np.array([1, 2, 4, 0, 1, 2])
        data = np.array([1, 2, 2, 1, 1, 2])
        unigram_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(3, 5)
        )
        # [[0, 1, 2, 0, 2]
        #  [1, 1, 0, 0, 0]
        #  [0, 0, 2, 0, 0]]

        result = _closest_ref_len(3, unigram_count_mtrx)
        expected = 2
        self.assertEqual(result, expected)

    def test_brevity_penalty_longer_references(self):
        result = _brevity_penalty(16, 12)
        expected = 1
        self.assertEqual(result, expected)

    def test_brevity_penalty_shorter_references(self):
        result = _brevity_penalty(9, 12)
        expected = 0.71653131
        self.assertAlmostEqual(result, expected)

    def test_modified_precision(self):
        # hypothesis
        row = np.array([0, 0, 0])
        col = np.array([1, 2, 4])
        data = np.array([2, 2, 1])
        hyp_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(1, 5)
        )
        # [[0, 2, 2, 0, 1]]

        # references
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 1, 2, 1, 2, 3])
        data = np.array([1, 1, 3, 1, 1, 2])
        ref_count_mtrx = sp.sparse.csr_matrix(
            (data, (row, col)), shape=(3, 5)
        )
        # [[1, 1, 0, 0, 0]
        #  [0, 0, 3, 0, 0]
        #  [0, 1, 1, 2, 0]]

        result = _modified_precision(
            hyp_count_mtrx, ref_count_mtrx
        )
        expected = (4, 6)
        self.assertEqual(result, expected)


class SubgraphDensity(unittest.TestCase):
    def test_no_instances(self):
        feat_mtrx = sp.sparse.coo_matrix([
        ])
        labels = np.array([
        ])
        with self.assertRaises(ScarceDataError):
            subgraph_density(feat_mtrx, labels)

    def test_some_class_counts_are_zero(self):
        feat_mtrx = sp.sparse.coo_matrix([
            [0.0, 0.0, 0.39881203, 0.0],
            [0.0, 0.19940602, 0.19940602, 0.48045301],
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.19940602, 0.0, 0.0, 0.0]
        ])
        # cosine similarities per class
        # [1.0]
        # [[1.0,    0.1909, 0.0   ]
        #  [0.1909, 1.0,    0.8457]
        #  [0.0,    0.8457, 1.0   ]]
        # []
        labels = np.array([
            [True, False, False],
            [False, True, False],
            [False, True, False],
            [False, True, False]
        ])

        result = subgraph_density(feat_mtrx, labels)
        expected = [None, 0.34557711, None]
        self.assertEqual(result, expected)

    def test_different_max_size_same_result(self):
        feat_mtrx = sp.sparse.coo_matrix([
            [0.0, 0.0, 0.39881203, 0.0],
            [0.0, 0.19940602, 0.19940602, 0.48045301],
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.19940602, 0.0, 0.0, 0.0]
        ])
        # cosine similarities per class
        # [1.0]
        # [[1.0,    0.1909, 0.0   ]
        #  [0.1909, 1.0,    0.8457]
        #  [0.0,    0.8457, 1.0   ]]
        labels = np.array([
            [True, False],
            [False, True],
            [False, True],
            [False, True]
        ])

        result1 = subgraph_density(
            feat_mtrx, labels
        )
        result2 = subgraph_density(
            feat_mtrx, labels, 3*10**-6
        )
        self.assertEqual(result1, result2)


class MinimumHellingerDistance(unittest.TestCase):
    def test_two_classes(self):
        count_mtrx = sp.sparse.coo_matrix([
            [2, 0, 0, 1, 2],
            [1, 2, 1, 0, 0]
        ])
        # ngram distribution per class
        # [0.4,  0.0, 0.0,  0.2, 0.4]
        # [0.25, 0.5, 0.25, 0.0, 0.0]
        one_hot_labels = np.array([
            [True, False],
            [False, True]
        ])

        result = minimum_hellinger_distance(
            [count_mtrx], one_hot_labels
        )
        expected = [0.8269052, 0.8269052]
        np.testing.assert_almost_equal(result, expected)

    def test_more_classes(self):
        count_mtrx = sp.sparse.coo_matrix([
            [0, 0, 3, 0],
            [0, 0, 0, 1],
            [2, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 2, 0, 0]
        ])
        # ngram distribution per class
        # [0.0, 0.0,  0.75, 0.25]
        # [0.6, 0.4,  0.0,  0.0 ]
        # [0.0, 0.75, 0.25, 0.0 ]
        one_hot_labels = np.array([
            [True, False, False],
            [True, False, False],
            [False, False, True],
            [False, False, True],
            [False, True, False],
            [False, True, False]
        ])

        result = minimum_hellinger_distance(
            [count_mtrx], one_hot_labels
        )
        expected = [0.7529855, 0.6725157, 0.6725157]
        np.testing.assert_almost_equal(result, expected)

    def test_more_ngram_sizes(self):
        uni_count_mtrx = sp.sparse.coo_matrix([
            [0, 0, 3, 0],
            [0, 0, 0, 2],
            [2, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 0, 0]
        ])
        bi_count_mtrx = sp.sparse.coo_matrix([
            [0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 2, 0, 0, 0, 0],
        ])
        # ngram distribution per class
        # unigram                      bigram
        # [0.25,  0.125,  0.375, 0.25][0.0, 0.2, 0.4, 0.0, 0.2, 0.2]
        # [0.125, 0.625,  0.25,  0.0 ][0.0, 0.4, 0.4, 0.2, 0.0, 0.0]
        one_hot_labels = np.array([
            [True, False],
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [False, True]
        ])
        result = minimum_hellinger_distance(
            [uni_count_mtrx, bi_count_mtrx], one_hot_labels
        )
        expected = [0.5252681, 0.5252681]
        np.testing.assert_almost_equal(result, expected)

    def test_some_class_count_is_zero(self):
        count_mtrx = sp.sparse.coo_matrix([
            [0, 0, 3, 0],
            [0, 0, 0, 1],
            [2, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 2, 0, 0]
        ])
        # ngram distribution per class
        # [0.0, 0.0,  0.75, 0.25]
        # [0.6, 0.4,  0.0,  0.0 ]
        # [0.0, 0.75, 0.25, 0.0 ]
        one_hot_labels = np.array([
            [True, False, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, False, True],
            [False, True, False, False],
            [False, True, False, False]
        ])

        result = minimum_hellinger_distance(
            [count_mtrx], one_hot_labels
        )
        expected = [0.75298557, 0.67251576, None, 0.67251576]
        self.assertEqual(result, expected)

    def test_some_word_count_is_zero(self):
        count_mtrx = sp.sparse.coo_matrix([
            [0, 0, 0, 3, 0],
            [0, 0, 0, 0, 1],
            [0, 2, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 2, 0, 0]
        ])
        # ngram distribution per class
        # [0.0, 0.0, 0.0,  0.75, 0.25]
        # [0.0, 0.6, 0.4,  0.0,  0.0 ]
        # [0.0, 0.0, 0.75, 0.25, 0.0 ]
        one_hot_labels = np.array([
            [True, False, False],
            [True, False, False],
            [False, False, True],
            [False, False, True],
            [False, True, False],
            [False, True, False]
        ])

        result = minimum_hellinger_distance(
            [count_mtrx], one_hot_labels
        )
        expected = [0.75298557, 0.67251576, 0.67251576]
        self.assertEqual(result, expected)


class GeometricSeparabilityIndex(unittest.TestCase):
    def test_no_instances(self):
        feat_mtrx = sp.sparse.coo_matrix([
        ])
        labels = np.array([
        ])
        with self.assertRaises(ScarceDataError):
            geometric_separability_index(feat_mtrx, labels)

    def test_some_class_no_instances(self):
        feat_mtrx = sp.sparse.coo_matrix([
            [0.0, 0.19940602, 0.19940602, 0.48045301],
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.19940602, 0.0, 0.0, 0.0]
        ])
        labels = np.array([
            [False, True],
            [False, True],
            [False, True]
        ])
        with self.assertRaises(ScarceDataError):
            geometric_separability_index(feat_mtrx, labels)

    def test_single_nn(self):
        feat_mtrx = sp.sparse.coo_matrix([
            [0.0, 0.0, 0.39881203, 0.0],
            [0.0, 0.19940602, 0.19940602, 0.48045301],
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.19940602, 0.0, 0.0, 0.0]
        ])
        # cosine similarities
        # [[1.0,    0.3579, 0.0,    0.0   ]
        #  [0.3579, 1.0,    0.1909, 0.0   ]
        #  [0.0,    0.1909, 1.0,    0.8457]
        #  [0.0,    0.0,    0.8457, 1.0   ]]
        labels = np.array([
            [True, False],
            [False, True],
            [False, True],
            [False, True]
        ])

        result = geometric_separability_index(
            feat_mtrx, labels
        )
        expected = [0.0, 0.66666667]
        self.assertEqual(result, expected)

    def test_several_nn(self):
        feat_mtrx = sp.sparse.coo_matrix([
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.0, 0.19940602, 0.19940602, 0.48045301],
            [0.0, 0.19940602, 0.19940602, 0.48045301]
        ])
        # cosine similarities
        # [[1.0,    1.0,    1.0,    0.1909, 0.1909]
        #  [1.0,    1.0,    1.0,    0.1909, 0.1909]
        #  [1.0,    1.0,    1.0,    0.1909, 0.1909]
        #  [0.1909, 0.1909, 0.1909, 1.0,    1.0   ]
        #  [0.1909, 0.1909, 0.1909, 1.0,    1.0   ]]
        labels = np.array([
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [True, False]
        ])
        result = geometric_separability_index(
            feat_mtrx, labels
        )
        expected = [0.33333333, 0]
        self.assertEqual(result, expected)

    def test_more_than_two_classes(self):
        feat_mtrx = sp.sparse.coo_matrix([
            [0, 2, 4, 1, 0],
            [1, 1, 2, 0, 0],
            [2, 0, 0, 2, 0],
            [1, 0, 4, 1, 2],
            [0, 2, 3, 1, 1],
            [1, 1, 0, 2, 3],
            [2, 3, 2, 0, 0],
            [1, 2, 4, 2, 1]
        ])
        # cosine similarities
        # [[1.    0.8908 0.1543 0.7909 0.9578 0.2253 0.7409 0.9415]
        # [0.8908 1.     0.2886 0.7833 0.8432 0.2108 0.8911 0.8807]
        # [0.1543 0.2886 1.     0.3015 0.1825 0.5477 0.3429 0.4160]
        # [0.7909 0.7833 0.3015 1.     0.8257 0.4954 0.5170 0.8780]
        # [0.9578 0.8432 0.1825 0.8257 1.     0.4666 0.7514 0.9621]
        # [0.2253 0.2108 0.5477 0.4954 0.4666 1.     0.3131 0.5063]
        # [0.7409 0.8911 0.3429 0.5170 0.7514 0.3131 1.     0.7610]
        # [0.9415 0.8807 0.4160 0.8780 0.9621 0.5063 0.7610 1.   ]]
        labels = np.array([
            [True, False, False],
            [False, False, True],
            [False, True, False],
            [False, True, False],
            [True, False, False],
            [True, False, False],
            [False, False, True],
            [False, False, True]
        ])
        result = geometric_separability_index(
            feat_mtrx, labels
        )
        expected = [0.33333333, 0.0, 0.66666667]
        self.assertEqual(result, expected)


    def test_different_max_size_same_result(self):
        feat_mtrx = sp.sparse.coo_matrix([
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.31605106, 0.19940602, 0.0, 0.0],
            [0.0, 0.19940602, 0.19940602, 0.48045301],
            [0.0, 0.19940602, 0.19940602, 0.48045301]
        ])
        # cosine similarities
        # [[1.0,    1.0,    1.0,    0.1909, 0.1909]
        #  [1.0,    1.0,    1.0,    0.1909, 0.1909]
        #  [1.0,    1.0,    1.0,    0.1909, 0.1909]
        #  [0.1909, 0.1909, 0.1909, 1.0,    1.0   ]
        #  [0.1909, 0.1909, 0.1909, 1.0,    1.0   ]]
        labels = np.array([
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [True, False]
        ])

        result1 = geometric_separability_index(
            feat_mtrx, labels
        )
        result2 = geometric_separability_index(
            feat_mtrx, labels, 8*10**-6
        )
        self.assertAlmostEqual(result1, result2)


class ImbalanceRatioTestCase(unittest.TestCase):
    def test_no_instances(self):
        one_hot_labels = np.array([
        ])
        with self.assertRaises(ScarceDataError):
            imbalance_ratio(one_hot_labels)

    def test_all_instances_same_class(self):
        one_hot_labels = np.array([
            [True, False, False],
            [True, False, False],
            [True, False, False]
        ])
        with self.assertRaises(ScarceDataError):
            imbalance_ratio(one_hot_labels)

    def test_some_class_counts_are_zero(self):
        one_hot_labels = np.array([
            [True, False, False],
            [False, True, False],
            [False, True, False],
            [True, False, False],
            [True, False, False],
            [True, False, False]
        ])
        result = imbalance_ratio(one_hot_labels)
        expected = [1.0, 0.5]
        self.assertEqual(result, expected)

    def test_balanced_classes(self):
        one_hot_labels = np.array([
            [True, False],
            [False, True],
            [True, False],
            [False, True]
        ])
        result = imbalance_ratio(one_hot_labels)
        expected = [1.0, 1.0]
        self.assertEqual(result, expected)


class MulticlassImbalanceRatioTestCase(unittest.TestCase):
    def test_no_instances(self):
        one_hot_labels = np.array([
        ])
        with self.assertRaises(ScarceDataError):
            multiclass_imbalance_ratio(one_hot_labels)

    def test_all_instances_same_class(self):
        one_hot_labels = np.array([
            [True, False, False],
            [True, False, False],
            [True, False, False]
        ])
        with self.assertRaises(ScarceDataError):
            multiclass_imbalance_ratio(one_hot_labels)

    def test_some_class_counts_are_zero(self):
        one_hot_labels = np.array([
            [True, False, False],
            [False, True, False],
            [False, True, False],
            [True, False, False],
            [True, False, False],
            [True, False, False]
        ])
        result = multiclass_imbalance_ratio(one_hot_labels)
        expected = 0.6
        self.assertAlmostEqual(result, expected)

    def test_balanced_classes(self):
        one_hot_labels = np.array([
            [True, False],
            [False, True],
            [True, False],
            [False, True]
        ])
        result = multiclass_imbalance_ratio(one_hot_labels)
        expected = 1
        self.assertAlmostEqual(result, expected)

    def test_imbalanced_classes(self):
        one_hot_labels = np.array([
            [True, False, False],
            [False, True, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [False, True, False],
            [True, False, False],
            [False, False, True],
        ])
        result = multiclass_imbalance_ratio(one_hot_labels)
        expected = 0.7
        self.assertAlmostEqual(result, expected)


class SubsetRepresentativityTestCase(unittest.TestCase):
    def test_empty_instances(self):
        count_mtrx = sp.sparse.coo_matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        one_hot_labels = np.array([
            [True, False],
            [True, False],
            [False, True],
            [False, True]
        ])
        with self.assertRaises(ScarceDataError):
            subset_representativity(count_mtrx, one_hot_labels)

    def test_no_instances(self):
        count_mtrx = sp.sparse.coo_matrix([
            [0, 2, 1, 1]
        ])
        one_hot_labels = np.array([
            [True, False]
        ])
        result = subset_representativity(count_mtrx, one_hot_labels)
        expected = [None, None]
        self.assertEqual(result, expected)

    def test_correct_input(self):
        count_mtrx = sp.sparse.coo_matrix([
            [1, 0, 1, 2, 1],
            [2, 3, 0, 2, 1],
        ])
        one_hot_labels = np.array([
            [True],
            [True]
        ])
        result = subset_representativity(count_mtrx, one_hot_labels)
        expected = [0.75]
        self.assertEqual(result, expected)


class NumberOfInstances(unittest.TestCase):
    def test_imbalanced_classes(self):
        one_hot_labels = np.array([
            [True, False, False],
            [False, True, False],
            [True, False, False],
            [True, False, False],
            [True, False, False],
            [False, True, False],
            [True, False, False],
            [False, False, True],
        ])
        result = number_of_instances(one_hot_labels)
        expected = [5, 2, 1]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
