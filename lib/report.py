"""report.py - create comprehensive report on all complexity metrics"""

import logging
import math

import pandas as pd
import scipy as sp
from tqdm import tqdm

from lib.metrics import (
    geometric_separability_index,
    imbalance_ratio,
    subgraph_density,
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


class ComplexityReport:
    fields = [
        "Self-BLEU",
        "Subgraph Density",
        "Minimum Hellinger Distance",
        "Geometric Separability Index",
        "Imbalance Ratio",
        "Subset Representativity",
        "Number of Classes",
        "Number of Instances"
    ]

    def __init__(self, class_names, name=None, info=None):
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

        """

        self.name = name
        if info is None:
            self.info = pd.DataFrame(columns=self.fields, index=class_names)
        else:
            self.info = info

    @classmethod
    def from_raw_data(cls, texts, labels, name=None):
        """Compute complexity measures from raw text.

        Args:
            texts (iterable):
                Holding raw text instances or file objects.
            labels (np.ndarray):
                One-hot encoded labels of shape (n_instance, n_class).

        Returns:
            ComplexityReport

        """
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
        numerical_labels, class_names = get_numerical_from_categorical(labels)
        one_hot_labels = get_one_hot_encoding_from_numerical(
            numerical_labels
        )

        # create empty report instance
        report = cls(class_names, name)
        # compute complexity metrics
        report._fill(
            unigram_counts, bigram_counts,
            tf_idf_feat, one_hot_labels
        )
        return report

    def _fill(self, unigram_counts, bigram_counts, tf_idf_feat, one_hot_labels):
        """Compute the set of available complexity measures."""

        LOG.info("----> Computing complexity metrics ...")

        with tqdm(total=8, leave=False) as pbar:

            pbar.set_description(f"{'Compute Self-Bleu ...':40}")
            self.info["Self-BLEU"] = self_bleu(
                [unigram_counts, bigram_counts], one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Subgraph Density ...':40}")
            self.info["Subgraph Density"] = subgraph_density(
                tf_idf_feat, one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Minimum Hellinger Distance ...':40}")
            self.info["Minimum Hellinger Distance"] = minimum_hellinger_distance(
                [unigram_counts, bigram_counts], one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Geometric Separability Index ...':40}")
            self.info["Geometric Separability Index"] = geometric_separability_index(
                tf_idf_feat, one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Imbalance Ratio ...':40}")
            self.info["Imbalance Ratio"] = imbalance_ratio(
                one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Subset Representativity ...':40}")
            self.info["Subset Representativity"] = subset_representativity(
                unigram_counts, one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Number of Classes ...':40}")
            self.info["Number of Classes"] = number_of_classes(
                one_hot_labels
            )
            pbar.update(1)
            pbar.set_description(f"{'Compute Number of Instances ...':40}")
            self.info["Number of Instances"] = number_of_instances(
                one_hot_labels
            )
            pbar.update(1)

