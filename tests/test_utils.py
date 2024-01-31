import numpy as np

from polygraph.utils import pad_with_Ns, reverse_complement


def test_reverse_complement():
    """
    Test reverse complement
    """
    assert reverse_complement("AGCA") == "TGCT"
    assert reverse_complement("NNNN") == "NNNN"


def test_pad_with_Ns():
    assert pad_with_Ns("AGCT", seq_len=6, end="both") == "NAGCTN"
    assert pad_with_Ns("AGCT", seq_len=6, end="left") == "NNAGCT"
    assert pad_with_Ns("AGCT", seq_len=6, end="right") == "AGCTNN"
    assert np.all(
        pad_with_Ns(["AGCT", "AGC", "CCCCC"], seq_len=None, end="right")
        == ["AGCTN", "AGCNN", "CCCCC"]
    )
