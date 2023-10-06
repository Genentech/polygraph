import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../sequence-evaluation'))
from utils import pad_with_Ns, rc


def test_rc():
    """
    Test reverse complement
    """
    assert rc("AGCA") == "TGCT"
    assert rc("NNNN") == "NNNN"


def test_pad_with_Ns():
    assert pad_with_Ns("AGCT", seq_len=6, end="both") == "NAGCTN"
    assert pad_with_Ns("AGCT", seq_len=6, end="left") == "NNAGCT"
    assert pad_with_Ns("AGCT", seq_len=6, end="right") == "AGCTNN"
    assert np.all(
        pad_with_Ns(["AGCT", "AGC", "CCCCC"], seq_len=None, end="right")
        == ["AGCTN", "AGCNN", "CCCCC"]
    )