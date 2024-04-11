import itertools

import editdistance
import numpy as np
import pandas as pd

# Constants

STANDARD_BASES = ["A", "C", "G", "T"]


def gc(seqs):
    """
    Calculate the GC fraction of a DNA sequence or list of sequences.

    Args:
        seqs (str, list, pd.DataFrame): A DNA sequence, list of sequences
            or dataframe containing sequences in the column "Sequence".

    Returns:
        (list, float): The fraction of each sequence comprised of G and C bases.

    """
    if isinstance(seqs, str):
        return float(seqs.count("G") + seqs.count("C")) / len(seqs)

    elif isinstance(seqs, list):
        return [gc(seq) for seq in list(seqs)]

    elif isinstance(seqs, pd.DataFrame):
        return gc(seqs.Sequence.tolist())

    else:
        raise TypeError("seqs must be a string, list or dataframe.")


def kmer_frequencies(seqs, k, normalize=False, genome="hg38"):
    """
    Get frequencies of all kmers of length k in a sequence or sequences.

    Args:
        seqs (str, list, pd.DataFrame): A DNA sequence, list of sequences
            or dataframe containing sequences in the column "Sequence".
        k (int): The k-mer length.
        normalize (bool, optional): Whether to normalize the k-mer counts
            by sequence length. Default is False.

    Returns:
        (pd.DataFrame): A dataframe of shape (kmers x sequences), containing
            the frequency of each k-mer in the sequence.
    """
    # Get all possible kmers
    kmers = ["".join(kmer) for kmer in itertools.product(STANDARD_BASES, repeat=k)]

    # Calculate kmer frequencies for a single sequence
    if isinstance(seqs, str):
        assert k <= len(
            seqs
        ), "k must be smaller than or equal to the length of the sequence"

        # Dictionary of all possible kmers
        output = {kmer: seqs.count(kmer) for kmer in kmers}

        # Make dataframe with kmers as rows
        output = pd.DataFrame.from_dict(output, orient="index")

        # Normalize
        if normalize:
            output[0] /= len(seqs) - k + 1

        return output

    # For multiple sequences, concatenate into a single dataframe
    elif isinstance(seqs, list):
        return pd.concat(
            [kmer_frequencies(seq, k, normalize) for seq in seqs], axis=1
        ).T.reset_index(drop=True)

    # For a dataframe, copy the index
    elif isinstance(seqs, pd.DataFrame):
        return kmer_frequencies(seqs.Sequence.tolist(), k, normalize).set_index(
            seqs.index
        )

    else:
        raise TypeError("seqs must be a string, list or dataframe.")


def unique_kmers(seq, k):
    """
    Get all unique kmers of length k that are present in a DNA sequence.

    Args:
        seq (str): the input DNA sequence
        k (int): length of k-mers to extract

    Returns:
        (set): a set containing the unique kmers extracted from
            the sequence.
    """
    assert k <= len(
        seq
    ), "k must be smaller than or equal to the length of the sequence"
    return set([seq[i : i + k] for i in range(len(seq) - k + 1)])


def kmer_positions(seq, kmer):
    """
    Return all the locations of a given k-mer in a DNA sequence

    Args:
        seq (str): the input DNA sequence
        kmer (str): the k-mer for which to search

    Returns:
        (np.array): a numpy array containing the positions of the kmer
    """
    kmers = np.array([seq[i : i + len(kmer)] for i in range(len(seq) - len(kmer) + 1)])
    return np.where(kmers == kmer)[0]


def _min_edit_distance(seq, reference_seqs):
    """
    Find the smallest edit distance between a sequence and a list of reference sequences

    Args:
        seq (str): A DNA sequence
        reference_seqs (list): A list of DNA sequences

    Returns:
        (int): edit distance between the sequence 'seq' and its closest
            neighbor in 'reference_seqs'.
    """
    return int(np.min([editdistance.eval(seq, rseq) for rseq in reference_seqs]))


def min_edit_distance(seqs, reference_seqs):
    """
    For each sequence in a list, find the smallest edit distance between that sequence
    and a list of reference sequences

    Args:
        seqs (list): List of sequences
        reference_seqs (list): List of sequences

    Returns:
        edit distance between each sequence in seqs and its closest reference sequence
    """
    return [_min_edit_distance(seq, reference_seqs) for seq in seqs]


def min_edit_distance_from_reference(seqs, reference_group, group_col="Group"):
    """
    For each sequence in non-reference groups, find the smallest edit distance
        between that sequence and the sequences in the reference group.

    Args:
        seqs (pd.DataFrame): Dataframe containing sequences in column "Sequence"
        reference_group (str): ID for the group to use as reference
        group_col (str): Name of the column containing group IDs

    Returns:
        edit (np.array): list of edit distance between each sequence and its closest
            reference sequence.
        Set to 0 for reference sequences
    """
    # List nonreference groups
    groups = seqs[group_col].unique()
    nonreference_groups = list(groups[groups != reference_group])

    # Create empty array
    edit = np.zeros(len(seqs), dtype=int)

    # Get reference sequences
    reference_seqs = seqs.Sequence[seqs[group_col] == reference_group].tolist()

    # Calculate distances
    for group in nonreference_groups:
        in_group = seqs[group_col] == group
        group_seqs = seqs.Sequence[in_group].tolist()
        edit[in_group] = min_edit_distance(group_seqs, reference_seqs)

    return edit


def groupwise_mean_edit_dist(seqs, group_col="Group"):
    """
    Calculate average edit distances between all groups of sequences
    """
    dist_matrix = np.zeros((len(seqs), len(seqs)))
    groups = seqs[group_col].unique()
    sequences = seqs.Sequence.tolist()

    # Calculate edit distance for each pair
    for i, s1 in enumerate(sequences):
        for j, s2 in enumerate(sequences):
            dist_matrix[i, j] = editdistance.eval(s1, s2)

    # Calculate average dist between groups
    group_dist = []
    for g1 in groups:
        for g2 in groups:
            dist = (
                dist_matrix[seqs[group_col] == g1, :][:, seqs[group_col] == g2]
                .mean()
                .mean()
            )
            group_dist.append((g1, g2, dist))

    group_dist = pd.DataFrame(group_dist, columns=["Group1", "Group2", "Edit"])
    group_dist = group_dist.pivot(index="Group1", columns="Group2", values="Edit")
    return group_dist


def bleu_similarity(seqs, reference_seqs, max_k=4):
    """
    Calculate the bleu similarity score between two sets of sequences.

    Args:
        seqs (list): List of DNA sequences
        reference_seqs (list): List of DNA sequences
        max_k (int): Highest k-mer length for calculation. All k-mers of
            length 1 to max_k inclusive will be considered.
    """
    from nltk.translate.bleu_score import corpus_bleu

    # Equal weight for each k
    weights = [1 / max_k] * max_k

    # Split into characters
    seqs = [[s for s in seq] for seq in seqs]
    reference_seqs = [[s for s in seq] for seq in reference_seqs]

    # Calculate score
    return corpus_bleu(reference_seqs, seqs, weights=weights)


def fastsk(seqs, k=5, m=2):
    """
    Compute a gapped k-mer kernel matrix for the given sequences using FastSK.

    Args:
        seqs (str, list, pd.DataFrame): A DNA sequence, list of sequences
            or dataframe containing sequences in the column "Sequence".
        k (int): k-mer length
        m (int): Number of mismatches allowed

    Returns:
        (np.array): Array of shape (n_seqs, n_seqs) containing the gapped k-mer kernel.
    """
    from fastsk import FastSK

    from polygraph.utils import integer_encode

    # Encode the input sequences
    inputs = integer_encode(seqs)

    # Compute kernel
    kernel = FastSK(g=k, m=m)
    kernel.compute_kernel(inputs, inputs)
    return np.array(kernel.get_train_kernel())


def ISM(seqs):
    """
    Perform in-silico mutagenesis on given DNA sequence(s)

    Args:
        seqs (str, list, pd.DataFrame): A DNA sequence, list of sequences
            or dataframe containing sequences in the column "Sequence".

    Returns:
        (list): A list of all possible single-base mutated sequences
            derived from the original sequences.
    """
    # ISM for a single sequence
    if isinstance(seqs, str):
        return list(
            np.concatenate(
                [
                    [seqs[:pos] + base + seqs[pos + 1 :] for base in STANDARD_BASES]
                    for pos in range(len(seqs))
                ]
            )
        )

    # Multiple sequences
    elif isinstance(seqs, list):
        return list(np.concatenate([ISM(seq) for seq in seqs]))

    # For a dataframe, copy the index
    elif isinstance(seqs, pd.DataFrame):
        return ISM(seqs.Sequence.tolist())

    else:
        raise TypeError("seqs must be a string, list or dataframe.")
