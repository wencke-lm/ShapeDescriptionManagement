"""metrics.py - provides metrics measuring classification difficulty"""

import logging
import math

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lib.exceptions import ScarceDataError
from lib.utils import blockwise_cosine_similarity


LOG = logging.getLogger(__name__)


# ============================================================================
#
# INTRA-CLASS DIVERSITY
#
# ============================================================================


def self_bleu(count_mtrx, labels, unigram_idx=0, n_references=10):
    """Compute intra-class self BLEU score for each class.

    Args:
        count_mtrx (list):
            Sequence of matrices (sp.sparse.coo_matrix)
            of word occurrence counts of shape (n_instance, n_voc).
            Each holding counts for a single ngram size.
        labels (np.ndarray):
            One-hot encoded gold labels of shape (n_instance, n_class).
        unigram_idx (Optional[int]):
            Index of unigram counts in <count_mtrx>. Defaults to 0.
        n_references (Optional[int]):
            Number of references to be sampled. If larger than number
            of instances, all instances are used as references,
            except the hypothesis. Defaults to 10.

    Returns:
        list: 
            Scores in range (0, 1] per class.
            Indices correspond to numerical class label. 
            Higher scores indicate lower intra-class diversity.

    """
    if labels.shape[0] == 0:
        raise ScarceDataError("The data set is empty.")
    if not count_mtrx:
        raise ScarceDataError("No count matrices have been passed.")

    # necessary for efficient row operations (slicing)
    count_mtrx = [mtrx.tocsr() for mtrx in count_mtrx]

    bleu_scores = []

    with tqdm(total=labels.shape[0], leave=False) as pbar:
        for i, mask in enumerate(labels.transpose()):
            # partition matrix by class
            cls_count_mtrx = [mtrx[mask] for mtrx in count_mtrx]
            # calculate intra-class self BLEU
            try:
                bleu_scores.append(
                    _self_bleu(
                        cls_count_mtrx, unigram_idx, n_references, pbar
                    )
                )
            except ScarceDataError as e:
                LOG.warning(e)
                bleu_scores.append(None)

        if all(map(lambda score: score is None, bleu_scores)):
            raise ScarceDataError(
                "All classes hold too few instances (< 2)."
            )

    return bleu_scores


def _self_bleu(count_mtrx, unigram_idx=0, n_references=10, pbar=None):
    """Compute the corpus-level self BLEU score.

    A BLEU score is calculated by treating every instance once
    as a hypothesis and randomly sampling a number n of strictly
    different instances as references.

    Args:
        count_mtrx (list):
            Sequence of matrices (sp.sparse.coo_matrix)
            of word occurrence counts of shape (n_instance, n_voc).
            Each holding counts for a single ngram size.
        unigram_idx (Optional[int]):
            Index of unigram counts in count_mtrx. Defaults to 0.
        n_references (Optional[int]):
            Number of references to be sampled. If larger than number
            of instances, all instances are used as reference.
        pbar (Optional[tqdm.std.tqdm]):
            A progress bar that will be updated if passed.

    Returns:
        float:
            Score in range (0, 1].
            Higher scores are indicative of
            less diversity between instances.

    """
    n_instance = count_mtrx[0].shape[0]

    if n_instance <= 1:
        raise ScarceDataError(
            "A class holds too few (< 2) instances."
        )

    hyp_lengths = 0
    ref_lengths = 0

    num = [0 for _ in count_mtrx]
    denom = [0 for _ in count_mtrx]

    for hyp_idx in range(n_instance):
        # retrieve reference indices
        refs_idx = np.array([
            idx for idx in range(n_instance) if idx != hyp_idx
        ])
        if n_instance > n_references:
            refs_idx = np.random.choice(
                refs_idx, n_references, replace=False
            )

        # every matrix holds counts for one ngram size
        for i, mtrx in enumerate(count_mtrx):
            mp_numerator, mp_denominator = _modified_precision(
                mtrx[hyp_idx], mtrx[refs_idx]
            )
            num[i] += mp_numerator
            denom[i] += mp_denominator

        # length is derived from raw unigram counts
        hyp_len = count_mtrx[unigram_idx][hyp_idx].sum()
        hyp_lengths += hyp_len
        ref_lengths += _closest_ref_len(
            hyp_len, count_mtrx[unigram_idx][refs_idx]
        )
        if pbar is not None:
            pbar.update(1)

    bp = _brevity_penalty(hyp_lengths, ref_lengths)
    w = 1/len(count_mtrx)
    score = math.fsum(w*math.log(n/d) for n, d in zip(num, denom))

    return round(bp*math.exp(score), 8)


def _closest_ref_len(hyp_len, references):
    """Find reference that is closest to hypothesis in length.

    If a longer and shorter reference exit that have the same
    difference in length to the hypothesis. One reference
    is randomly sampled.

    Args:
        hyp_len (int): Length of the hypothesis.
        references (sp.sparse.csr_matrix):
            Raw unigram occurrence counts of
            shape (n_reference, n_unigram).

    Returns:
        int: Length of reference that is closest in length.

    """
    # holds at most 2 elements
    closest_len = set()
    diff_to_closest = float("inf")

    for ref in references:
        ref_len = ref.sum()
        diff = abs(hyp_len - ref_len)

        if diff < diff_to_closest:
            closest_len = {ref_len}
            diff_to_closest = diff
        elif diff == diff_to_closest:
            closest_len.add(ref_len)

    return np.random.choice(list(closest_len))


def _brevity_penalty(hyp_len, closest_ref_len):
    """Compute the previty penalty.

    Penalize a hypothesis when its length is shorter
    than the closest reference length.

    ... bp = min(1, e^(1 - (len_ref/len_hyp)))

    Args:
        hyp_len (int): Length of a hypothesis OR
            sum of all hypotheses' lengths in corpus.
        closest_ref_len (int):
            Length of reference closest in length to hypothesis OR
            sum of closest reference lengths for all hypotheses.

    Returns:
        float: Penalty in range [0, 1]. 1 if closest
            references are not shorter than hypotheses.

    """
    if hyp_len == 0:
        return 0
    return min(1, np.exp(1 - closest_ref_len / hyp_len))


def _modified_precision(hypothesis, references):
    """Compute the modified ngram precision.

    The normal precision has the disadvantage that a hypothesis
    that contains a word from the reference several times has very
    high precision. One therefore imposes the restriction that a
    reference word can be used to match at most one hypothesis word.

    Let C(x, y) be the number of times that x occurs in y:
    ... match = sum_ngram min(C(ngram, hyp), max_j C(ngram, ref_j))
    ... p_m = match / (sum_ngram C(ngram, hyp))

    Args:
        hypothesis (sp.sparse.csr_matrix):
            Raw ngram occurrence counts of shape (1, n_ngram).
            Holding counts for a single ngram size.
        references (sp.sparse.csr_matrix):
            Raw ngram occurrence counts of shape (n_reference, n_gram).
            Holding counts for a single ngram size.

    Returns:
        tuple: numerator, denominator
            Precision is calculated as a fraction of both.

    """
    # keep only those ngram counts that occur in the hypothesis
    references = references.tocsc()[:, hypothesis.indices]

    # for every reference subtract its ngram counts from the hypothesis' ones
    # for every ngram in hypothesis retrieve its minimum unmatched occurrences
    unmatched = (hypothesis.data - references).min(axis=0)
    unmatched.clip(min=0, out=unmatched)

    denominator = hypothesis.sum()
    numerator = denominator - unmatched.sum()

    # smoothing from Chin-Yew Lin & Franz Josef Och (2004)
    return numerator + 1, denominator + 1


def subgraph_density(feat_mtrx, labels, max_size=200):
    """Compute mean subgraph density.

    A feature encoded dataset is represented as a graph by
    translating instances to nodes and their pairwise
    similarities to edge weights between those nodes.

    Graph density measures, how connected a graph is,
    compared to how connected it might be.
    Let V be the graph's nodes and A_ij the weight of the
    edge connecting node i and j:
    ... D = 2*( sum_i=1^|V| sum_j=i+1^|V| A_ij ) / ( |V|*(|V|-1) )

    Args:
        labels (np.ndarray):
            One-hot encoded gold labels of shape (n_instance, n_class).
        feat_mtrx (sp.sparse.coo_matrix):
            Feature encoded dataset of shape (n_instance, n_feat).
        max_size (Optional[int]):
            Upper-bound for the number of cells in million in a
            computed similarity matrix. Larger datasets will be
            evaluated in blocks. Defaults to 200.

    Returns:
        list: 
            Density in range [0, 1] per class.
            Indices correspond to numerical class label.
            Higher values are indicative of more densely
            connected, less diverse nodes within a class.

    """
    if labels.shape[0] == 0:
        raise ScarceDataError("The data set is empty.")
    # necessary for efficient row operations (slicing)
    feat_mtrx = feat_mtrx.tocsr()

    densities = []

    for mask in labels.transpose():
        # partition matrix by class
        cls_mtrx = feat_mtrx[mask]
        n_instance, _ = cls_mtrx.shape

        if n_instance < 2:
            LOG.warning("A class holds too few (< 2) instances.")
            densities.append(None)
            continue

        densities.append(0)
        for start_idx, adj_mtrx in blockwise_cosine_similarity(
            cls_mtrx, max_size
        ):
            densities[-1] += np.tril(adj_mtrx, -1-start_idx).sum()

        densities[-1] /= n_instance*(n_instance-1)/2
        densities[-1] = round(densities[-1], 8)

    return densities


# ============================================================================
#
# INTER-CLASS INTERFERENCE
#
# ============================================================================


def minimum_hellinger_distance(count_mtrx, labels):
    """Compute the minimum pairwise distance between classes.

    Classes are modelled as ngram distributions. Each class
    is compared to all other classes and its distances to the
    closest neighbour class calculated.

    Args:
        count_mtrx (list):
            Sequence of matrices (sp.sparse.coo_matrix) of word
            occurrence counts of shape (n_instance, n_voc).
            Each holding counts for a single ngram size.
        labels (np.ndarray):
            One-hot encoded gold labels of shape (n_instance, n_class).

    Returns:
        list: 
            Minimum distance in the range [0, 1] per class.
            Indices correspond to numerical class label.
            Lower values indicate that at least two
            classes are very similar to each other.

    """
    # necessary for efficient column operations (column-sum)
    count_mtrx = [mtrx.tocsc() for mtrx in count_mtrx]

    cls_distr = []
    min_distance = [1 for _ in range(labels.shape[1])]

    for k, mask in enumerate(labels.transpose()):
        # partition matrix by class
        self_cls_distr = [
            mtrx[mask].sum(axis=0, dtype=np.float32) for mtrx in count_mtrx
        ]
        self_cls_distr = [
            distr/distr.sum() for distr in self_cls_distr if distr.sum() != 0
        ]

        if len(self_cls_distr) != len(count_mtrx):
            LOG.warning(
                "At least one instance holds no ngrams of specified ngram size."
                "Check your data and remove empty strings."
            )
            min_distance[k] = None
            continue

        for l, other_cls_distr in cls_distr:
            distances = []
            # average the hellinger distances acroos ngram sizes
            for i, distr in enumerate(self_cls_distr):
                distances.append(
                    _hellinger_distance(other_cls_distr[i], distr)
                )
            distance = sum(distances)/len(distances)
            min_distance[k] = round(min(min_distance[k], distance), 8)
            min_distance[l] = round(min(min_distance[l], distance), 8)

        cls_distr.append((k, self_cls_distr))

    return min_distance


def _hellinger_distance(p, q):
    """Compute hellinger distance between probability dist. P and Q.

    ... H(P, Q) = sqrt( sum_i=1^k(sqrt(p_i) - sqrt(q_i))^2 ) / sqrt(2)

    Args:
        p (np.ndarray): Discrete prob. dist. of shape (n_ngram,).
        q (np.ndarray): Discrete prob. dist. of shape (n_ngram,).

    Returns:
        float: Single distance value in the range [0, 1].
            Maximum distance 1 is achieved when one distribution
            assigns zero probability to every outcome that the
            other distribution assigns a positive probability to.

    """
    if ((abs(p.sum() - 1) > 1e-5 and p.sum() != 0)
            or (abs(q.sum() - 1) > 1e-5 and q.sum() != 0)):
        raise ValueError("Probability distributions need to be passed.")

    summe = np.square(np.sqrt(p) - np.sqrt(q)).sum()
    return summe**0.5 / 2**0.5


def geometric_separability_index(feat_mtrx, labels, max_size=200):
    """Compute the geometric separability index.

    The proportion of instances that have a nearest neighbour(NN)
    of the same class. Let |I| be the number of instances:
    ... GSI = sum_i^|I| 1[class(i) == class(i_NN)] / |I|

    Note:
        If an instance has several NN of the same distance, all
        are considered. Instead of adding 1 for a match, the
        proportion (0<p<1) of NN that matches is added.

    Args:
        labels (np.ndarray):
            One-hot encoded gold labels of shape (n_instance, n_class).
        feat_mtrx (sp.sparse.coo_matrix):
            Feature encoded dataset of shape (n_instance, n_feat).
        max_size (int): Upper-bound for the number of cells
            in million in a computed similarity matrix.
            Larger datasets will be evaluated in blocks.

    Returns:
        float: Proportions in the range [0, 1] per class.
            Indices correspond to numerical class label.
            If this measure is high, it is likely that
            a clear boundary between classes can be found.
            However, the opposite is not the case.
            If instances of different classes are arranged
            in straight parallel lines, the proportion may
            be low, despite the existence of a clear boundary.

    """
    if (labels.sum(axis=0) == 0).any():
        raise ScarceDataError("At least one class holds no instances.")

    nn_match = [0 for _ in range(labels.shape[1])]

    # transform from one-hot to integer representation
    _, labels = np.where(labels)
    # necessary for efficient row operations (slicing)
    feat_mtrx = feat_mtrx.tocsr()

    for start_idx, adj_mtrx in blockwise_cosine_similarity(
        feat_mtrx, max_size
    ):
        # remove similarities of an instance to itself
        np.fill_diagonal(adj_mtrx[start_idx:], -1)

        nn_simil = np.amax(adj_mtrx, axis=0)
        nn_idx, instance_idx = np.nonzero(adj_mtrx == nn_simil)

        matches = (labels[nn_idx] == labels[start_idx+instance_idx])
        # normalize by number of nearest neighbour per instance
        normalized_matches = matches/np.bincount(instance_idx)[instance_idx]

        for label, match in zip(labels[nn_idx], normalized_matches):
            nn_match[label] += match

    return list(np.around(np.array(nn_match) / np.bincount(labels), 8))


# ============================================================================
#
# CLASS IMBALANCE
#
# ============================================================================


def imbalance_ratio(labels):
    label_counts = labels.sum(axis=0)

    if np.count_nonzero(label_counts) < 2:
        raise ScarceDataError("Too few (< 2) classes included.")

    label_counts = label_counts[np.nonzero(label_counts)]
    imbalance_ratio = label_counts/label_counts.max()

    return list(np.around(imbalance_ratio, 8))



def multiclass_imbalance_ratio(labels):
    """Compute the inverse imbalance ratio.

    Let C be the number of classes, |Ic| the number of instances
    in class c and |I| the total number of instances.
    Imbalance ratio is defined as:
    ... IR = [( C - 1 ) / ( C )] * sum_c=1^C [|Ic| / (|I|???|Ic|)]

    Args:
        labels (np.ndarray): One-hot encoded gold
            labels of shape (n_instance, n_class).

    Returns:
        float: Inverse imbalance ratio in range [1, ???].
            Higher values indicate a higher degree of balance.
            A perfectly balanced dataset achieves a value of 1.

    """
    label_counts = labels.sum(axis=0)

    if np.count_nonzero(label_counts) < 2:
        raise ScarceDataError("Too few (< 2) classes included.")

    factor = (len(label_counts) - 1) / len(label_counts)
    summe = sum(label_counts / (len(labels) - label_counts))
    imbalance_ratio = factor * summe

    return 1 / imbalance_ratio


# ============================================================================
#
# DATA SPARSITY
#
# ============================================================================


def subset_representativity(count_mtrx, labels, steps=100):
    """Compute a measure of subset representativity.

    Each class is split into two non-overlapping subsets
    of equal size. One of those two sets is understood
    to be new incoming data. Then, the number of unique
    words that occur within this part and have also been
    observed for the other part is computed.

    We define the ratio of seen types:
    ... r_seen:total = count(tok_seen) / count(tok_total)

    Args:
        count_mtrx (sp.sparse.coo_matrix):
            Word occurence counts as positive integers in
            a matrix of shape (n_instance, n_voc).
        labels (np.ndarray):
            One-hot encoded gold labels of shape (n_instance, n_class).
        steps (int):
            Number of times to repeat the random split process.

    Returns:
        list: 
            Ratio seen types : total types in range [0, 1] per class.
            Indices correspond to numerical class label.
            A sparse class, where one part provides no information
            about the other part, will achieve values close to 0.

    """
    # necessary for efficient row operations (slicing)
    count_mtrx = count_mtrx.tocsr()

    seen_type_ratio = []

    for mask in labels.transpose():
        # partition matrix by class
        cls_mtrx = count_mtrx[mask]

        if (mask.sum() < 2):
            LOG.warning("A class holds too few (< 2) instances.")
            seen_type_ratio.append(None)
            continue

        seen_type_ratio_cls = 0

        for _ in range(steps):
            old_count_mtrx, new_count_mtrx = train_test_split(
                cls_mtrx, train_size=0.5
            )
            old_word_occ_counts = old_count_mtrx.sum(axis=0)
            new_word_occ_counts = new_count_mtrx.sum(axis=0)

            seen_type_count = np.count_nonzero(
                np.logical_and(
                    old_word_occ_counts > 0,
                    new_word_occ_counts > 0
                )
            )
            new_type_count = np.count_nonzero(new_word_occ_counts)

            if new_type_count == 0:
                raise ScarceDataError(
                    "At least one instance holds no tokens."
                    "Check your data and remove empty strings."
                )
            else:
                seen_type_ratio_cls += seen_type_count / new_type_count

        seen_type_ratio.append(round(seen_type_ratio_cls/steps, 8))

    return seen_type_ratio


# ============================================================================
#
# OTHERS
#
# ============================================================================


def number_of_classes(labels):
    """Get the number of unique classes in a data set.

    Args:
        labels (np.ndarray): One-hot encoded gold
            labels of shape (n_instance, n_class).

    Returns:
        int: Number of unique classes.

    """
    return labels.shape[1]


def number_of_instances(labels):
    """Get the number of instances per class.

    Args:
        labels (np.ndarray):
            One-hot encoded gold labels of
            shape (n_instance, n_class).

    Returns:
        list:
            Number of instances per class.
            Indices correspond to numerical class label.

    """
    return list(labels.sum(axis=0))
