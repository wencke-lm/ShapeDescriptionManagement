"""report_test.py - verify methods for complexity report creation"""

import os
import sys
import unittest

# in order to access module from sister directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from lib.report import ComplexityReport


class ReportTestCase(unittest.TestCase):
    def test_compute_all_metrics(self):
        texts = [
            "this cat sleeps",
            "this dog sleeps cutely",
            "a cat",
            "a a dog"
        ]
        labels = [
            "cat",
            "dog",
            "cat",
            "dog"
        ]
        result = ComplexityReport.from_raw_data(texts, labels)
        # might change due to random component
        result.pop("Subset Representativity")

        expected = {
            "Mean Self-BLEU": 0.41721988,
            "Mean Subgraph Density": 0.07192994,
            "Minimum Hellinger Distance": 0.82515094,
            "Geometric Separability Index": 0.0,
            "Imbalance Ratio": 1.0,
            "Number of Classes": 2,
            "Number of Instances": 4
        }
        self.assertEqual(result, expected)

    def test_normalize_metrics(self):
        texts = [
            "this cat sleeps",
            "this dog sleeps cutely",
            "a cat",
            "a a dog"
        ]
        labels = [
            "cat",
            "dog",
            "cat",
            "dog"
        ]
        result = ComplexityReport.from_raw_data(texts, labels)
        # might change due to random component
        result.pop("Subset Representativity")
        # normalize
        result.normalize()

        expected = {
            "Mean Self-BLEU": 0.41721988,
            "Mean Subgraph Density": 0.07192994,
            "Minimum Hellinger Distance": 0.82515094,
            "Geometric Separability Index": 0.0,
            "Imbalance Ratio": 1.0,
            "Number of Classes": 0.5,
            "Number of Instances": 0.00057757
        }
        self.assertEqual(result, expected)
