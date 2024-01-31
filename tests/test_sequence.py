import numpy as np

from polygraph.sequence import (
    gc,
    kmer_frequencies,
    kmer_positions,
    unique_kmers,
)


def test_gc():
    """
    Test GC content calculation.
    """
    assert gc("AGCGN") == 3 / 5
    assert gc(["GC", "AG"]) == [1.0, 0.5]


def test_kmer_frequencies():
    seqs = "AGCTAAAA"
    expected_output = np.array([2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    assert np.all(
        kmer_frequencies(seqs, k=2, normalize=False)[0].tolist() == expected_output
    )
    assert np.all(
        kmer_frequencies(seqs, k=2, normalize=True)[0].tolist() == expected_output / 7
    )


def test_unique_kmers():
    seqs = "AGCTAA"
    assert set(unique_kmers(seqs, k=2)) == set(["AG", "GC", "CT", "TA", "AA"])


def test_kmer_positions():
    seqs = "AGCTAAA"
    assert np.all(kmer_positions(seqs, "AA") == np.array([4, 5]))
