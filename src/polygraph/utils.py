import numpy as np
import pandas as pd

# Constants
RC_HASH = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
}
base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}


def make_ids(seqs):
    """
    Assign a unique index to each row of a dataframe

    Args:
        seqs (pd.DataFrame): Pandas dataframe

    Returns:
        seqs (pd.DataFrame): Modified database containing unique indices.
    """
    seqs["SeqID"] = [f"seq_{i}" for i in range(len(seqs))]
    return seqs.set_index("SeqID")


def get_lens(seqs):
    """
    Calculate the lengths of given DNA sequences.

    Args:
        seqs (str, list, pd.DataFrame): A DNA sequence, list of sequences
            or dataframe containing sequences in the column "Sequence".

    Returns:
        (int, list): length of each sequence
    """
    if isinstance(seqs, str):
        return len(seqs)

    elif isinstance(seqs, list):
        return [len(seq) for seq in seqs]

    elif isinstance(seqs, pd.DataFrame):
        return get_lens(seqs.Sequence.tolist())

    else:
        raise TypeError("seqs must be a string, list or dataframe.")


def check_equal_lens(seqs):
    """
    Given sequences, check whether they are all of equal length.

    Args:
        seqs (list, pd.DataFrame): Either a list of DNA sequences,
            or a dataframe containing DNA sequences in the column
            "Sequence".

    Returns:
        (bool): whether the sequences are all equal in length.

    """
    return len(set(get_lens(seqs))) == 1


def pad_with_Ns(seqs, seq_len=None, end="both"):
    """
    Pads a sequence with Ns at the desired end until it reaches
    `seq_len` in length.

    If seq_len is not provided, it is set to the length of
    the longest sequence.

    Args:
        seqs (str, list, pd.DataFrame): DNA sequence, list of sequences
            or dataframe containing sequences in the column "Sequence".
        seq_len (int): Length upto which to pad each sequence

    Returns:
        (str, list): Padded sequences of length seq_len
    """

    # Get seq_len
    seq_len = seq_len or np.max(get_lens(seqs))
    # print(f"Padding all sequences to length {seq_len}")

    # Pad a single sequence
    if isinstance(seqs, str):
        padding = seq_len - len(seqs)
        if padding > 0:
            if end == "both":
                start_padding = padding // 2
                end_padding = padding - start_padding
                return "N" * start_padding + seqs + "N" * end_padding
            elif end == "left":
                return "N" * padding + seqs
            elif end == "right":
                return seqs + "N" * padding
        else:
            return seqs

    # Pad multiple sequences
    elif isinstance(seqs, list):
        return [pad_with_Ns(seq, seq_len=seq_len, end=end) for seq in seqs]

    elif isinstance(seqs, pd.DataFrame):
        return pad_with_Ns(seqs.Sequence.tolist(), seq_len=seq_len, end=end)

    else:
        raise TypeError("seqs must be a string, list or dataframe.")


def reverse_complement(seqs):
    """
    Reverse complement DNA sequences

    Args:
        seqs (str, list, pd.DataFrame): seqs (str, list): A DNA sequence, list of
            sequences or dataframe containing sequences in the column "Sequence".

    Returns:
        (str, list): reverse complemented sequences

    """
    if isinstance(seqs, str):
        return "".join([RC_HASH[base] for base in reversed(seqs)])
    elif isinstance(seqs, list):
        return [reverse_complement(seq) for seq in seqs]
    elif isinstance(seqs, pd.DataFrame):
        return reverse_complement(seqs.Sequence.tolist())
    else:
        raise TypeError("seqs must be a string, list or dataframe.")


def integer_encode(seqs):
    """
    Encode DNA sequence(s) as a numpy array of integers.

    Args:
        seqs (str, list, pd.DataFrame): seqs (str, list): A DNA sequence, list of
            sequences or dataframe containing sequences in the column "Sequence".

    Returns:
        (np.array): A 1-D or 2-D array containing the sequences encoded as integers.
    """
    if isinstance(seqs, str):
        return np.array([base_to_idx[x] for x in seqs])

    elif isinstance(seqs, list):
        return np.array([[base_to_idx[x] for x in seq] for seq in seqs])

    elif isinstance(seqs, pd.DataFrame):
        return integer_encode(seqs.Sequence.tolist())

    else:
        raise TypeError("Input should be a string, list of strings, or dataframe")
