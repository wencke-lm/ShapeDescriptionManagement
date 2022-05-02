"""report.py - create comprehensive report on all complexity metrics"""

import logging
import math

import scipy as sp
from tqdm import tqdm

from lib.metrics import (
    geometric_separability_index,
    imbalance_ratio,
    mean_subgraph_density,
    minimum_hellinger_distance,
    number_of_classes,
    number_of_instances,
    self_bleu,
    subset_representativity,
)
from lib.utils import (
    count_ngrams,
    get_numerical_from_categorical,
    get_one_hot_encoding_from_numerical,
    tf_idf
)


LOG = logging.getLogger(__name__)


class ComplexityReport(dict):
    fields = [
        "Mean Self-BLEU",
        "Mean Subgraph Density",
        "Minimum Hellinger Distance",
        "Geometric Separability Index",
        "Imbalance Ratio",
        "Subset Representativity",
        "Number of Classes",
        "Number of Instances"
    ]

    def __init__(self, name=None, precomputed=None):
        """Collection of complexity measures.

        Attributes:
            unigram_counts (sp.sparse.coo_matrix):
                Raw unigram counts of shape (n_instance, n_unigram).
            bigram_counts (sp.sparse.coo_matrix):
                Raw bigram counts of shape (n_instance, n_bigram).
            tf_idf_feat (sp.sparse.coo_matrix):
                tf-idf matrix of shape (n_instance, n_term).
            one_hot_labels (np.ndarray):
                Matrix of shape (n_instance, n_class).
            normalisation_manual (dict):
                Metric name mapped to a three-member boolean list.
                Each member represents one type of normalisation.
                0: log10(orig_value + 10)
                1: 1 / orig_value
                2: 1 - orig_value

        """
        if precomputed is not None:
            if any(map(lambda m: m not in self.fields, precomputed)):
                raise ValueError
            self.update(precomputed)

        self.name = name

        # set normalisation instructions
        self.normalisation_manual = {
            metric: [False, False, False, False] for metric in self.fields
        }
        self.normalisation_manual["Number of Classes"] = [
            False, True, False
        ]
        self.normalisation_manual["Number of Instances"] = [
            True, True, True
        ]

    @classmethod
    def from_raw_data(cls, texts, labels, name=None):
        """Compute complexity measures from raw text.

        Args:
            texts (iterable):
                Holding raw text instances or file objects.
            labels (np.ndarray):
                One-hot encoding of shape (n_instance, n_class).

        Returns:
            ComplexityReport

        """
        # create empty class instance
        report = cls(name=name)

        # preprocess textual data
        LOG.info("----> Extracting ngram counts ...")
        unigram_counts = count_ngrams(texts, n=1)
        bigram_counts = count_ngrams(texts, n=2)

        LOG.info("----> Converting counts to tf-idf representation ...")
        tf_idf_feat = tf_idf(
            sp.sparse.hstack((unigram_counts, bigram_counts))
        )

        # preprocess categorical labels
        LOG.info("----> Converting labels to one-hot encoding ...")
        one_hot_labels = get_one_hot_encoding_from_numerical(
            get_numerical_from_categorical(labels)[0]
        )

        # compute complexity metrics
        report._fill(
            unigram_counts, bigram_counts,
            tf_idf_feat, one_hot_labels
        )
        return report

    def _fill(self, unigram_counts, bigram_counts, tf_idf_feat, one_hot_labels):
        """Compute the set of available complexity metrics."""

        LOG.info("----> Computing complexity metrics ...")

        with tqdm(total=8, leave=False) as pbar:
            pbar.set_description(f"{'Compute Mean Self-Bleu ...':40}")
            self["Mean Self-BLEU"] = self_bleu(
                [unigram_counts, bigram_counts], one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Mean Subgraph Density ...':40}")
            self["Mean Subgraph Density"] = mean_subgraph_density(
                tf_idf_feat, one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Minimum Hellinger Distance ...':40}")
            self["Minimum Hellinger Distance"] = minimum_hellinger_distance(
                [unigram_counts, bigram_counts], one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Geometric Separability Index ...':40}")
            self["Geometric Separability Index"] = geometric_separability_index(
                tf_idf_feat, one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Imbalance Ratio ...':40}")
            self["Imbalance Ratio"] = imbalance_ratio(
                one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Subset Representativity ...':40}")
            self["Subset Representativity"] = subset_representativity(
                unigram_counts, one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Number of Classes ...':40}")
            self["Number of Classes"] = number_of_classes(
                one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Number of Instances ...':40}")
            self["Number of Instances"] = number_of_instances(
                one_hot_labels
            )
            pbar.update(1)

        # round obtained values
        for key, value in self.items():
            self[key] = round(value, 8)

    def normalize(self):
        """Inplace normalisation of complexity metrics.

        Metrics are pushed into the range [0, 1],
        with lower values indicating a higher complexity.

        """

        # check for normalisation instructions
        unknown_metrics = {
            m for m in self if m not in self.normalisation_manual
        }

        if unknown_metrics:
            raise KeyError(
                "Cannot normalize unknown metrics: "
                f"{', '.join(unknown_metrics)}"
            )

        # normalize
        for name, value in self.items():
            if self.normalisation_manual[name][0]:
                value = math.log(value + 100, 100)
            if self.normalisation_manual[name][1]:
                value = 1 / value
            if self.normalisation_manual[name][2]:
                value = 1 - value

            self[name] = round(value, 8)
