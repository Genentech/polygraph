import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../sequence-evaluation'))
from motifs import process_motif, motif_frequencies, scan_seqs, motif_combinations
from input import read_jaspar


jaspar_motifs = read_jaspar('files/jaspar.txt')[0]
processed_jaspar_motifs = motifs.process_motif(jaspar_motif)


def test_process_motif():
    pass


def test_motif_frequencies():
    pass


def test_scan_seqs():
    pass


def test_motif_combinations():
    pass