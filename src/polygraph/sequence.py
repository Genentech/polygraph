import itertools

import editdistance
import numpy as np
import pandas as pd

STANDARD_BASES = ["A", "C", "G", "T"]


def gc(seqs, sequence_col="Sequence"):
    """
    Calculate the GC fraction of a DNA sequence or list of sequences.

    Args:
        seqs (list, str): The DNA sequences to calculate the GC content of.

    Returns:
        (list, float): The fraction of the sequence comprised of G and C bases.

    """
    if isinstance(seqs, str):
        return float(seqs.count("G") + seqs.count("C")) / len(seqs)

    elif (isinstance(seqs, list)) or (isinstance(seqs, pd.Series)):
        return [gc(seq) for seq in list(seqs)]

    elif isinstance(seqs, pd.DataFrame):
        return [gc(seq) for seq in list(seqs[sequence_col])]


def kmer_frequencies(seqs, k, normalize=False, genome="hg38"):
    """
    Get frequencies of all kmers of length k in a sequence.

    Args:
        seqs (pd.DataFrame): DNA sequences, as intervals, strings, or tensors.
        k (int): The length of the k-mer.
        normalize (bool, optional): Whether to normalize the histogram so that
            the values sum to 1.
        Default is False.

    Returns:
        (pd.DataFrame): A dataframe of shape (kmers x sequences), containing
        the frequency of each k-mer in the sequence.
    """
    # Get all possible kmers
    kmers = ["".join(kmer) for kmer in itertools.product(STANDARD_BASES, repeat=k)]

    if isinstance(seqs, str):
        assert k <= len(
            seqs
        ), "k must be smaller than or equal to the length of the sequence"

        # Dictionary of all possible kmers
        output = {"".join(kmer): seqs.count(kmer) for kmer in kmers}

        # Make dataframe with kmers as rows
        output = pd.DataFrame.from_dict(output, orient="index")

        # Normalize
        if normalize:
            output[0] /= len(seqs) - k + 1

        return output

    if isinstance(seqs, pd.DataFrame):
        ids = seqs.SeqID.tolist()
        seqs = seqs.Sequence.tolist()
    else:
        ids = range(len(seqs))

    # For multiple sequences, concatenate into a single dataframe
    if isinstance(seqs, list):
        output = pd.concat(
            [kmer_frequencies(seq, k, normalize) for seq in seqs], axis=1
        )

    output.columns = ids
    return output.T


def unique_kmers(seq, k):
    """
    Get unique kmers of length k in a sequence.

    Args:
        seq (str): the input sequence
        k (int): length of kmers to extract

    Returns:
        (np.array): a numpy array containing the unique kmers extracted from
            the sequence.
    """
    assert k <= len(
        seq
    ), "k must be smaller than or equal to the length of the sequence"
    return np.unique([seq[i : i + k] for i in range(len(seq) - k + 1)])


def kmer_positions(seq, kmer):
    """
    Return the locations of a kmer in a sequence

    Args:
        seq (str): the input sequence
        kmer (str): the kmer to search for

    Returns:
        (np.array): a numpy array containing the positions of the kmer
    """
    k = np.array([seq[i : i + len(kmer)] for i in range(len(seq) - len(kmer) + 1)])
    return np.where(k == kmer)[0]


def _min_edit_distance(seq, reference_seqs):
    """
    Find the smallest edit distance between a sequence and a list of reference sequences

    Args:
        seq (str): Sequence
        reference_seqs (list): List of sequences

    Returns:
        edit distance between the sequence and its closest reference sequence
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


def min_edit_distance_from_reference(df, reference_group, group_col="Group"):
    """
    For each sequence in non-reference groups, find the smallest edit distance
        between that sequence and the sequences in the reference group.

    Args:
        df (pd.DataFrame): Dataframe containing sequences in column "Sequence"
        reference_group (str): ID for the group to use as reference
        group_col (str): Name of the column containing group IDs

    Returns:
        edit (np.array): list of edit distance between each sequence and its closest
            reference sequence.
        Set to 0 for reference sequences
    """
    # List nonreference groups
    groups = df[group_col].unique()
    nonreference_groups = list(groups[groups != reference_group])

    # Create empty array
    edit = np.zeros(len(df), dtype=int)

    # Get reference sequences
    reference_seqs = df.Sequence[df[group_col] == reference_group].tolist()

    # Calculate distances
    for group in nonreference_groups:
        in_group = df[group_col] == group
        group_seqs = df.Sequence[in_group].tolist()
        group_edit = min_edit_distance(group_seqs, reference_seqs)
        edit[in_group] = group_edit

    return edit


def bleu_similarity(seqs, reference_seqs, max_k=4):
    """
    bleu similarity score between two sets of sequences.

    Args:
        seqs (list): List of sequences
        reference_seqs (list): List of sequences
        max_k (int): Highest k-mer length for calculation. All k-mers of
        length 1 to max_k inclusive will be considered.
    """
    from nltk.translate.bleu_score import corpus_bleu

    weights = [1 / max_k] * max_k

    # Split into characters
    seqs = [[s for s in seq] for seq in seqs]
    reference_seqs = [[s for s in seq] for seq in reference_seqs]

    # Calculate score
    return corpus_bleu(reference_seqs, seqs, weights=weights)
