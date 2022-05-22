"""report_test.py - verify methods for complexity report creation"""

import os
import sys
import unittest

import pandas as pd

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
        report = ComplexityReport.from_raw_data(texts, labels).info
        # Subset Representativity might change due to random component
        report.drop(columns=["Subset Representativity"], inplace=True)

        result_cat = report.loc["cat"].tolist()
        expected_cat = [
            0.47809144, 0.093642, 0.82515094, 0.0, 1.0, 2.0, 2.0
        ]
        self.assertEqual(result_cat, expected_cat)

        result_dog = report.loc["dog"].tolist()
        expected_dog = [
            0.35634832, 0.05021789, 0.82515094, 0.0, 1.0, 2.0, 2.0
        ]
        self.assertEqual(result_dog, expected_dog)

    def test_missing_metric(self):
        texts = [
            "this cat sleeps",
            "this dog sleeps cutely",
            "a a dog"
        ]
        labels = [
            "cat",
            "dog",
            "dog"
        ]
        report = ComplexityReport.from_raw_data(texts, labels).info

        result_cat = pd.isna(report.loc["cat"]).tolist()
        expected_cat = [
            True, True, False, False, False, True, False, False
        ]
        self.assertEqual(result_cat, expected_cat)
