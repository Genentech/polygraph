import os

from polygraph.input import read_jaspar, read_pfms, read_seqs

cwd = os.path.realpath(os.path.dirname(__file__))


def test_read_seqs():
    df = read_seqs(os.path.join(cwd, "files", "seqs.tsv"), sep="\t")
    assert df.Sequence.tolist() == ["AAAA", "AAAC"]
    assert df.Group.tolist() == ["group1", "group2"]
    assert df.SeqID.tolist() == ["seq_0", "seq_1"]


def test_read_jaspar():
    motifs = read_jaspar(os.path.join(cwd, "files", "jaspar.txt"))
    assert isinstance(motifs, list)
    assert len(motifs) == 2


def test_read_pfms():
    motifs = read_pfms(os.path.join(cwd, "files", "motifs"))
    assert isinstance(motifs, list)
    assert len(motifs) == 2
