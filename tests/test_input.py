import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../sequence-evaluation'))
from input import read_seqs, read_jaspar, read_pfms


def test_read_seqs():
    df = read_seqs('files/seqs.tsv', sep='\t')
    assert df.Sequence.tolist() == ['AAAA', 'AAAC']
    assert df.Group.tolist() == ['group1', 'group2']
    assert df.SeqID.tolist() == ['seq0', 'seq1']
    

def test_read_jaspar():
    motifs = read_jaspar('files/jaspar.txt')
    assert isinstance(motifs, list)
    assert len(motifs) == 2


def test_read_pfms():
    motifs = read_pfms('files/motifs')
    assert isinstance(motifs, list)
    assert len(motifs) == 2