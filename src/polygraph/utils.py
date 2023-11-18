import numpy as np
import torch


RC_HASH = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
}


def get_unique_lens(seqs):
    """
    Given sequences, return their lengths.

    Args:
        seqs (list, pd.DataFrame): DNA sequences or intervals

    Returns:
        (list): length of each sequence
    """
    return [len(seq) for seq in seqs]


def check_equal_lens(seqs):
    """
    Given sequences, check whether they are all of equal length

    Args:
        seqs (list, pd.DataFrame): DNA sequences or intervals

    Returns:
        (bool): whether the sequences are all equal

    """
    return len(set(get_unique_lens(seqs))) == 1


def pad_with_Ns(seqs, seq_len=None, end="both"):
    """
    Pads a sequence with Ns at desired end to reach `seq_len`
    If seq_len is not provided, it is set to the length of
    the longest sequence.

    Args:
        seqs (str, list): DNA sequences
        seq_len (int): length to pad to

    Returns:
        (str, list): Padded sequences of length seq_len
    """
    if seq_len is None:
        seq_len = np.max(get_unique_lens(seqs))
        print("Padding all sequences to length {}".format(seq_len))

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

    elif isinstance(seqs, list):
        return [pad_with_Ns(seq, seq_len=seq_len, end=end) for seq in seqs]


def resize(seqs, seq_len):
    """
    Resize DNA sequences from center.

    Args:
        strings(str, list): DNA sequences
        seq_len (int): Desired length

    Returns:
        (str, list): Resized sequences
    """
    if isinstance(seqs, str):
        if len(seqs) >= seq_len:
            start = (len(seqs) - seq_len) // 2
            return seqs[start : start + seq_len]
        else:
            return pad_with_Ns(seqs, seq_len, end="both")
    else:
        return [resize_strings(seq, seq_len=seq_len) for seq in seqs]


def rc(seqs):
    """
    Reverse complement input sequences

    Args:
        seqs (str, list, torch.Tensor): DNA sequences as strings, indices or one-hot

    Returns:
        reverse complemented sequences in the same format

    """
    if isinstance(seqs, str):
        return "".join([RC_HASH[base] for base in reversed(seqs)])
    else:
        return ["".join([RC_HASH[base] for base in reversed(seq)]) for seq in seqs]

