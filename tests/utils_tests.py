"""utils_test.py - verify methods for data transformation"""

import os
import sys
import unittest

import numpy as np
import scipy as sp

# in order to access module from sister directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from lib.utils import (
    count_ngrams,
    tf_idf,
    get_numerical_from_categorical,
    get_one_hot_encoding_from_numerical
)


class UtilsTestCase(unittest.TestCase):
    def test_get_numerical_from_categorical(self):
        result = get_numerical_from_categorical(
            ["cat", "dog", "cat"]
        )
        expected = (
            np.array([0, 1, 0]), ["cat", "dog"]
        )
        np.testing.assert_array_equal(result[0], expected[0])
        self.assertEqual(result[1], expected[1])

    def test_get_one_hot_encoding_from_numerical(self):
        result = get_one_hot_encoding_from_numerical(
            [0, 1, 2, 1]
        )
        expected = np.array([
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, True, False]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_count_ngrams(self):
        result = count_ngrams(
            ["this cat", "a cat", "a a dog"]
        ).toarray()
        expected = np.array([
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [2, 0, 1, 0]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_tf_idf(self):
        result = tf_idf(sp.sparse.coo_matrix([
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [2, 0, 1, 0]
        ])).toarray()
        expected = np.array([
            [0.0, 0.0, 0.0, 0.281047],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.281047, 0.0]
        ])
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
