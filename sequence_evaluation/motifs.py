import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.decomposition import NMF
import MOODS.scan
import MOODS.tools
from statsmodels.stats.multitest import fdrcorrection
import itertools
from stats import *


def process_motif(motif, pseudocounts=1e-2, p=1e-3):
    """
    Compute log odds and threshold for a PWM
    """
    motif.log_odds = list(motif.counts.normalize(pseudocounts=pseudocounts).log_odds().values())
    motif.threshold = MOODS.tools.threshold_from_p(motif.log_odds, MOODS.tools.flat_bg(4), p)
    return motif


def moods_scan(seq, motifs):
    """
    Scan a sequence with a list of motifs using MOODS
    """
    return MOODS.scan.scan_dna(
        seq, 
        [motif.log_odds for motif in motifs],
        MOODS.tools.flat_bg(4),
        [motif.threshold for motif in motifs], 7)


def _motif_count(seq, motifs, normalize=True):
    """
    Count the number of occurrences of motifs in a sequence
    """
    div = 1
    if normalize:
        div = len(seq)
    return [len(r)/div for r in moods_scan(seq, motifs)]


def motif_frequencies(df, motifs, normalize=True, num_workers=8):
    """
    Count frequency of occurrence of motifs in a list of sequences
    """
    print("Processing motifs")
    motifs = [process_motif(motif) for motif in motifs]

    print("Scanning")
    with mp.Pool(num_workers) as pool:
        items = zip(df.Sequence.tolist(), [motifs]*len(df), [normalize]*len(df))
        cts = pool.starmap(_motif_count, items)

    print("Assembling count matrix")
    cts = pd.DataFrame(np.vstack(cts))
    cts.index = df.SeqID.tolist()
    cts.columns = [motif.name for motif in motifs]
    return cts


def _moods_results_to_df(motif, results):
    """
    Convert MOODS scan results to dataframe
    """
    df = pd.DataFrame.from_dict({
        "MotifID": motif.name, "score": [r.score for r in results], "start": [r.pos for r in results],
    })
    df['end'] = df.start + motif.length
    return df


def _scan_seq(seq, motifs, seq_id=None):
    """
    Scan a sequence with motifs using MOODS and return a dataframe
    """
    results = moods_scan(seq, motifs)
    results = pd.concat([_moods_results_to_df(motif, res) for motif, res in zip(motifs, results)])
    results['SeqID'] = seq_id
    return results


def scan_seqs(df, motifs, p=1e-3, pseudocounts=1e-2, num_workers=8, group_col='Group'):
    """
    Scan sequences with motifs using MOODS and return a dataframe
    """
    
    print("Processing motifs")
    motifs = [process_motif(motif) for motif in motifs]

    print("Scanning")
    with mp.Pool(num_workers) as pool:
        items = zip(df.Sequence.tolist(), [motifs]*len(df), df.SeqID.tolist())
        sites = pool.starmap(_scan_seq, items)

    sites = pd.concat(sites).reset_index(drop=True)
    sites = sites.merge(df[['SeqID', group_col]], on="SeqID", how="left")
    return sites


def nmf(counts, seqs, reference_group, group_col='Group', n_components=10):
    """
    Perform NMF on motif count table
    """

    # Run NMF
    model = NMF(n_components=n_components, init='random', random_state=0)

    # Format W and H matrices
    W = pd.DataFrame(model.fit_transform(counts.values))
    H = pd.DataFrame(model.components_)
    factors = [f'factor_{i}' for i in range(n_components)]
    W.index = counts.index
    W.columns = factors
    H.index = factors
    H.columns = counts.columns

    # Significance testing
    W[group_col] = seqs[group_col].tolist()
    
    res = pd.DataFrame()
    for col in W.columns[:-1]:
        factor_res = groupwise_mann_whitney(W, val_col=col, reference_group=reference_group, group_col=group_col)
        factor_res['factor'] = col
        res = pd.concat([res, factor_res])
    
    # FDR correction
    res['padj'] = fdrcorrection(res.pval)[1]

    return W, H, res


def motif_combinations(counts, seqs, reference_group, group_col='Group', min_group_freq=10):
    """
    Count occurences of pairwise combinations of motifs and compare between groups
    """

    print("Listing motif combinations")
    # Get set of motifs present in each sequence
    motif_combinations = counts.apply(lambda row: set(counts.columns[np.where(row > 0)[0]]), axis=1).reset_index()
    motif_combinations.columns = ['SeqID', 'motifs']
    motif_combinations = motif_combinations.merge(seqs[['SeqID', group_col]], on='SeqID')

    # Get pairwise combinations present in each sequence
    motif_combinations['combination'] = motif_combinations.motifs.apply(lambda x: list(itertools.combinations(x, 2)))
    motif_combinations = motif_combinations[['SeqID', 'combination', 'Group']].explode('combination')

    print("Making count matrix")
    # Count number of sequences in which each motif combination is present
    cts = motif_combinations[['combination', 'Group']].value_counts().reset_index()
    
    print("Filtering")
    # Drop rare combinations 
    comb_max = cts.groupby('combination')['count'].max()
    sel_comb = cts.combination[cts.combination.isin(comb_max[comb_max > min_group_freq].index)].tolist()
    motif_combinations = motif_combinations[motif_combinations.combination.isin(sel_comb)]
    cts = cts[cts.combination.isin(sel_comb)]

    print("Significance testing")
    # Significance testing
    df = seqs[['SeqID', 'Group']].copy()

    res = pd.DataFrame()
    
    for comb in cts.combination:
        seqs_with_comb = motif_combinations.SeqID[motif_combinations.combination == comb].tolist()
        df['has_comb'] = df.SeqID.isin(seqs_with_comb)
        curr_res = groupwise_fishers(df, reference_group=reference_group, val_col='has_comb', reference_val=None, group_col=group_col).reset_index()
        curr_res['combination'] = [comb]*len(curr_res)
        res = pd.concat([res, curr_res])
    
    # FDR correction
    res['padj'] = fdrcorrection(res.pval)[1]
    
    return res.reset_index(drop=True)