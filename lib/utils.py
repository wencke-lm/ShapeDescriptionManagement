"""utils.py - provides helper functions for data transformation"""

import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_numerical_from_categorical(vec):
    """Transform categorical sequence to integer encoding.

    Args:
        vec (Sequence): Any categorical sequence.

    Returns:
        tuple: np.ndarray, list
            Numeric array of shape (n_instance, ).
            Categorical labels at their assigned numeric position.

    Example:
        >>> get_numerical_from_categorical(["cat", "dog", "cat"])
        (array([0, 1, 0]), ["cat", "dog"])

    """
    cat_to_idx = dict()
    num_vec = [cat_to_idx.setdefault(cat, len(cat_to_idx)) for cat in vec]
    cat_fields = sorted(cat_to_idx, key=lambda cat: cat_to_idx[cat])

    return np.array(num_vec), cat_fields


def get_one_hot_encoding_from_numerical(vec):
    """Transform integer sequence to one-hot encoding.

    Args:
        vec (sequence): Any integer sequence.

    Returns:
        np.ndarray: Matrix of shape (n_instance, n_class).

    Example:
        >>> get_one_hot_encoding_from_numerical([0, 1, 2, 1])
        array([[ True, False, False],
               [False,  True, False],
               [False, False,  True],
               [False,  True, False]])

    """
    one_hot = np.full((len(vec), max(vec) + 1), False)
    one_hot[np.arange(len(vec)), vec] = True

    return one_hot


def count_ngrams(texts, n=1):
    """Counts the occurrences of ngrams per instance.

    Punctuation and capitalization is ignored.

    Args:
        texts (iterable): Holding raw text instances or file objects.
        n (int): Value of n for which ngrams should be extracted.

    Returns:
        sp.sparse.coo_matrix:
            Raw ngram occurrence counts of shape (n_instance, n_ngram).

    Example:
        >>> count_ngrams(
        ...     ['this cat', 'a cat', 'a a dog']
        ... ).toarray()
        array([[0, 1, 0, 1],
               [1, 1, 0, 0],
               [2, 0, 1, 0]])

    """
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(n, n))
    return cv.fit_transform(texts).tocoo()


def tf_idf(count_mtrx):
    """Transform a term frequency matrix to a tf-idf matrix.

    We compute the tf-idf for a term t in an instance i of a dataset D:
    ... tf-idf(t, i) = tf(t, i) * idf(t, D)

    Log normalized variants are used for both tf and idf.
    Let count(t, i) be the number of occurrences of a term t in i.
    ... tf(t, i) = log [ count(t, i) + 1 ],

    Let the nominator be the number of instances in total and
    the denominator the number of instances where term t occurs.
    ... idf(t, D) = log [ |I| / (1 + |{i in I : t in i}|) ],

    Args:
        count_mtrx (sp.sparse.coo_matrix): Raw word occurrence
            counts of shape (n_instance, n_term).

    Returns:
        sp.sparse.coo_matrix:
            tf-idf matrix of the shape (n_instance, n_term).

    """
    # necessary for efficient column operations (idf calculation)
    count_mtrx = count_mtrx.tocsc()

    row_n, _ = count_mtrx.get_shape()
    if row_n == 0:
        raise ValueError("Your term frequency matrix may not be empty.")

    tf = np.log(count_mtrx.data + 1)
    idf = np.atleast_2d(np.log(row_n / (count_mtrx.getnnz(axis=0) + 1)))

    # necessary for efficient operations (tf*idf multiplication)
    count_mtrx = count_mtrx.tocoo()

    # all empty cells have tf-idf of zero because their tf is zero
    count_mtrx.data = np.multiply(tf, idf[:, count_mtrx.col].ravel())

    return count_mtrx


def blockwise_cosine_similarity(feat_mtrx, max_size):
    """Compute cosine similarity in blocks that fit into memory.

    Args:
        feat_mtrx (sp.sparse.coo_matrix): Feature encoded
            dataset of shape (n_instance, n_feat).
        max_size (int): Upper-bound for the number of cells
            in million in a computed similarity matrix.
            Similarities for larger datasets are evaluated in blocks.

    Yields:
        (int, sp.sparse.csr_matrix):
            Start index of the instances included in the block.
            Similarity matrix of shape (n_instance, n_block) between
            the whole dataset and the block of instances.

    """
    n_instance, _ = feat_mtrx.shape
    step = int(max(min(max_size*10**6 // n_instance, n_instance), 1))

    for start in range(0, n_instance, step):
        yield start, cosine_similarity(
            feat_mtrx, feat_mtrx[start:start+step]
        )
