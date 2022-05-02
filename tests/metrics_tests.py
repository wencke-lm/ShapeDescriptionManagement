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
    infrequent_type_ratio,
    imbalance_ratio,
    mean_subgraph_density,
    minimum_hellinger_distance,
    self_bleu,
    subset_representativity,
    type_token_ratio,
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
        expected = 0.75
        self.assertEqual(result, expected)

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
        self.assertAlmostEqual(result, expected)

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
        expected = 4, 6
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

    def test_self_bleu_averaged_intra_class_unigrams_bigrams(self):
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
        expected = 0.4172198831
        self.assertAlmostEqual(result, expected)

    def test_self_bleu_averaged_intra_class_only_unigrams(self):
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
        expected = 0.39772727
        self.assertAlmostEqual(result, expected)


class MeanSubgraphDensity(unittest.TestCase):
    def test_no_instances(self):
        feat_mtrx = sp.sparse.coo_matrix([
        ])
        labels = np.array([
        ])
        with self.assertRaises(ScarceDataError):
            mean_subgraph_density(feat_mtrx, labels)

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

        result = mean_subgraph_density(feat_mtrx, labels)
        expected = 0.6727886
        self.assertAlmostEqual(result, expected)

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

        result1 = mean_subgraph_density(
            feat_mtrx, labels
        )
        result2 = mean_subgraph_density(
            feat_mtrx, labels, 3*10**-6
        )
        self.assertAlmostEqual(result1, result2)


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
        expected = 0.82690521
        self.assertAlmostEqual(result, expected)

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
        expected = 0.67251575
        self.assertAlmostEqual(result, expected)

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
        expected = 0.52526811
        self.assertAlmostEqual(result, expected)


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
        expected = 0.67251575
        self.assertAlmostEqual(result, expected)

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
        expected = 0.67251575
        self.assertAlmostEqual(result, expected)


class GeometricSeparabilityIndex(unittest.TestCase):
    def test_no_instances(self):
        feat_mtrx = sp.sparse.coo_matrix([
        ])
        labels = np.array([
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
        expected = 0.5
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
        expected = 0.2
        self.assertEqual(result, expected)

    def test_different_max_size_same_result_(self):
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
        expected = 0.6
        self.assertAlmostEqual(result, expected)

    def test_balanced_classes(self):
        one_hot_labels = np.array([
            [True, False],
            [False, True],
            [True, False],
            [False, True]
        ])
        result = imbalance_ratio(one_hot_labels)
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
        result = imbalance_ratio(one_hot_labels)
        expected = 0.7
        self.assertAlmostEqual(result, expected)


class DistinctTotalWordRatioTestCase(unittest.TestCase):
    def test_no_instances(self):
        count_mtrx = sp.sparse.coo_matrix([
        ])
        with self.assertRaises(ScarceDataError):
            type_token_ratio(count_mtrx)

    def test_empty_instances(self):
        count_mtrx = sp.sparse.coo_matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        with self.assertRaises(ScarceDataError):
            type_token_ratio(count_mtrx)

    def test_some_word_count_is_zero(self):
        count_mtrx = sp.sparse.coo_matrix([
            [1, 0, 2, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 2]
        ])
        result = type_token_ratio(count_mtrx)
        expected = 0.4
        self.assertEqual(result, expected)

    def test_sparse(self):
        count_mtrx = sp.sparse.coo_matrix([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        result = type_token_ratio(count_mtrx)
        expected = 1.0
        self.assertEqual(result, expected)

    def test_saturated(self):
        count_mtrx = sp.sparse.coo_matrix([
            [1, 2, 0],
            [1, 0, 3],
            [0, 0, 2],
            [1, 1, 0],
            [2, 2, 0],
        ])
        result = type_token_ratio(count_mtrx)
        expected = 0.2
        self.assertEqual(result, expected)


class InfrequentTypeRatioTestCase(unittest.TestCase):
    def test_no_instances(self):
        count_mtrx = sp.sparse.coo_matrix([
        ])
        with self.assertRaises(ScarceDataError):
            infrequent_type_ratio(count_mtrx)

    def test_empty_instances(self):
        count_mtrx = sp.sparse.coo_matrix([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        with self.assertRaises(ScarceDataError):
            infrequent_type_ratio(count_mtrx)

    def test_some_word_count_is_zero(self):
        count_mtrx = sp.sparse.coo_matrix([
            [1, 0, 2, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 2]
        ])
        result = infrequent_type_ratio(count_mtrx, delta=2)
        expected = 0.5
        self.assertEqual(result, expected)

    def test_sparse(self):
        count_mtrx = sp.sparse.coo_matrix([
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        result = infrequent_type_ratio(count_mtrx, delta=2)
        expected = 1.0
        self.assertEqual(result, expected)

    def test_saturated(self):
        count_mtrx = sp.sparse.coo_matrix([
            [1, 2, 0],
            [1, 0, 3],
            [0, 0, 2],
            [1, 1, 0],
            [2, 2, 0],
        ])
        result = infrequent_type_ratio(count_mtrx, delta=2)
        expected = 0.0
        self.assertEqual(result, expected)


class SubsetRepresentativityTestCase(unittest.TestCase):
    def test_no_instances(self):
        count_mtrx = sp.sparse.coo_matrix([
        ])
        one_hot_labels = np.array([
        ])
        with self.assertRaises(ScarceDataError):
            subset_representativity(count_mtrx, one_hot_labels)

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
        expected = 0.75
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
