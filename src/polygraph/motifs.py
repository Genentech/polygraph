import itertools
import multiprocessing as mp

import MOODS.scan
import MOODS.tools
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from statsmodels.stats.multitest import fdrcorrection

from polygraph.sequence import reverse_complement
from polygraph.stats import groupwise_fishers, groupwise_mann_whitney


def process_motif(motif, pseudocounts=1e-2, p=1e-3):
    """
    Compute log odds and threshold for a PWM.

    Args:
        motif (Bio.motif): Biopython motif object
        pseudocounts (float): Pseudocount to add to matrix before log odds calculation
        p (float): p-value for motif match threshold

    Returns
        motif (Bio.motif): Processed motif object, containing attributes `log_odds`
            and `threshold`.
    """
    # Calculate log odds (PWM) matrix using pseudocount
    motif.log_odds = list(
        motif.counts.normalize(pseudocounts=pseudocounts).log_odds().values()
    )

    # Calculate p-value threshold using MOODS
    motif.threshold = MOODS.tools.threshold_from_p(
        motif.log_odds, MOODS.tools.flat_bg(4), p
    )
    return motif


def moods_scan(seq, motifs):
    """
    Scan a sequence with a list of motifs using MOODS

    Args:
        seq (str): DNA sequence
        motifs (list): List of Bio.motif objects

    Returns:
        MOODS scan results
    """
    return MOODS.scan.scan_dna(
        seq,
        [motif.log_odds for motif in motifs],
        MOODS.tools.flat_bg(4),
        [motif.threshold for motif in motifs],
        7,
    )


def _motif_count(seq, motifs, normalize=True):
    """
    Count the number of occurrences of motifs in a sequence
    """
    div = 1
    if normalize:
        div = len(seq)
    return [len(r) / div for r in moods_scan(seq, motifs)]


def motif_frequencies(
    seqs, motifs, normalize=True, num_workers=8, pseudocounts=1e-2, p=1e-3, rc=False
):
    """
    Count frequency of occurrence of motifs in a list of sequences

    Args:
        seqs (pd.DataFrame): Pandas dataframe containing DNA sequences.
        motifs (list): List of Bio.motif objects
        normalize (bool): Whether to normalize the resulting count matrix
            to correct for sequence length
        num_workers (int): Number of parallel workers for scanning
        pseudocounts (float): Pseudocount to add to matrix before log odds calculation
        p (float): p-value for motif match threshold
        rc (bool): Whether to also scan the reverse complement sequence

    Returns:
        cts (pd.DataFrame): Count matrix with rows = sequences and columns = motifs
    """
    print("Processing motifs")
    motifs = [process_motif(motif, pseudocounts=pseudocounts, p=p) for motif in motifs]

    print("Scanning")
    with mp.Pool(num_workers) as pool:
        items = zip(
            seqs.Sequence.tolist(), [motifs] * len(seqs), [normalize] * len(seqs)
        )
        cts = pool.starmap(_motif_count, items)

    if rc:
        # Reverse complement sequences
        seqs["Sequence_rc"] = [
            reverse_complement(seq) for seq in seqs.Sequence.tolist()
        ]

        print("Scanning reverse strand sequences")
        with mp.Pool(num_workers) as pool:
            items = zip(
                seqs.Sequence_rc.tolist(), [motifs] * len(seqs), [normalize] * len(seqs)
            )
            cts_rev = pool.starmap(_motif_count, items)

    print("Assembling count matrix")
    cts = pd.DataFrame(np.vstack(cts))
    cts.index = seqs.SeqID.tolist()
    cts.columns = [motif.name for motif in motifs]

    if rc:
        cts_rev = pd.DataFrame(np.vstack(cts_rev))
        cts_rev.index = seqs.SeqID.tolist()
        cts_rev.columns = [motif.name + "_rev" for motif in motifs]
        cts.columns = [col + "_fwd" for col in cts.columns]
        cts = pd.concat([cts, cts_rev], axis=1)

    return cts


def _moods_results_to_df(motif, results):
    """
    Convert MOODS scan results to dataframe
    """
    df = pd.DataFrame.from_dict(
        {
            "MotifID": motif.name,
            "score": [r.score for r in results],
            "start": [r.pos for r in results],
        }
    )
    df["end"] = df.start + motif.length
    return df


def _scan_seq(seq, motifs, seq_id=None):
    """
    Scan a sequence with motifs using MOODS and return a dataframe
    """
    results = moods_scan(seq, motifs)
    results = pd.concat(
        [_moods_results_to_df(motif, res) for motif, res in zip(motifs, results)]
    )
    results["SeqID"] = seq_id
    return results


def scan_seqs(
    seqs,
    motifs,
    num_workers=8,
    group_col="Group",
    pseudocounts=1e-2,
    p=1e-3,
    rc=False,
):
    """
    Scan sequences with motifs using MOODS and return a dataframe

    Args:
        seqs (pd.DataFrame): Pandas dataframe containing DNA sequences.
        motifs (list): List of Bio.motif objects
        num_workers (int): Number of parallel workers for scanning
        group_col (str): Name of column in `seqs` that contains group IDs
        pseudocounts (float): Pseudocount to add to matrix before log odds calculation
        p (float): p-value for motif match threshold
        rc (bool): Whether to also scan the reverse complement sequence

    Returns:
        sites (pd.DataFrame): Pandas dataframe containing site positions
    """
    print("Processing motifs")
    motifs = [process_motif(motif, pseudocounts=pseudocounts, p=p) for motif in motifs]

    print("Scanning forward strand sequences")
    with mp.Pool(num_workers) as pool:
        items = zip(seqs.Sequence.tolist(), [motifs] * len(seqs), seqs.SeqID.tolist())
        sites = pool.starmap(_scan_seq, items)

    # Construct dataframe of results
    sites = pd.concat(sites).reset_index(drop=True)
    sites = sites.merge(seqs[["SeqID", group_col]], on="SeqID", how="left")

    if rc:
        # Reverse complement sequences
        seqs["Sequence_rc"] = [
            reverse_complement(seq) for seq in seqs.Sequence.tolist()
        ]

        print("Scanning reverse strand sequences")
        with mp.Pool(num_workers) as pool:
            items = zip(
                seqs.Sequence_rc.tolist(), [motifs] * len(seqs), seqs.SeqID.tolist()
            )
            sites_rev = pool.starmap(_scan_seq, items)

        # Construct dataframe of results
        sites_rev = pd.concat(sites_rev).reset_index(drop=True)
        seqs["len"] = seqs.Sequence.apply(len)
        sites_rev = sites_rev.merge(
            seqs[["SeqID", group_col, "len"]], on="SeqID", how="left"
        )

        # Correct positions
        start_coords = sites_rev["len"] - sites_rev["end"]
        end_coords = sites_rev["len"] - sites_rev["start"]

        sites_rev["start"] = start_coords
        sites_rev["end"] = end_coords

        sites_rev = sites_rev.drop(columns=["len"])

        # Combine
        sites["strand"] = "fwd"
        sites_rev["strand"] = "rev"
        sites = pd.concat([sites, sites_rev])

        # Add strand to motif_id
        sites["MotifID"] = sites["MotifID"] + "_" + sites["strand"]

    return sites


def nmf(counts, seqs, reference_group, group_col="Group", n_components=10):
    """
    Perform NMF on motif count matrix

    Args:
        counts (pd.DataFrame): motif count matrix where rows are sequences and columns
            are motifs.
        seqs (pd.DataFrame): pandas dataframe containing DNA sequences.
        reference_group (str): ID for the group to use as reference
        group_col (str): Name of the column in `seqs` containing group IDs
        n_components (int): Number of components or factors to extract using NMF

    Returns:
        W (pd.DataFrame): Pandas dataframe of size sequences x factors, containing
            the contribution of each factor to each sequence.
        H (pd.DataFrame): Pandas dataframe of size factors x motifs, containing the
            contribution of each motif to each factor.
        res (pd.DataFrame): Pandas dataframe containing the FDR-corrected significance
            testing results for factor contribution between groups.
    """
    # Run NMF
    model = NMF(n_components=n_components, init="random", random_state=0)

    # Obtain W and H matrices
    W = pd.DataFrame(model.fit_transform(counts.values))  # seqs x factors
    H = pd.DataFrame(model.components_)  # factors x motifs

    # Format W and H matrices
    factors = [f"factor_{i}" for i in range(n_components)]
    W.index = counts.index
    W.columns = factors
    H.index = factors
    H.columns = counts.columns

    # Add group IDs to W
    W[group_col] = seqs[group_col].tolist()

    # Significance testing between groups
    res = pd.DataFrame()
    for col in W.columns[:-1]:
        # For each factor, test whether its abundance differs between groups
        factor_res = groupwise_mann_whitney(
            W, val_col=col, reference_group=reference_group, group_col=group_col
        )
        factor_res["factor"] = col
        res = pd.concat([res, factor_res])

    # FDR correction
    res["padj"] = fdrcorrection(res.pval)[1]

    return W, H, res


def motif_combinations(
    counts,
    seqs,
    reference_group,
    group_col="Group",
    min_group_freq=10,
    min_group_prop=None,
):
    """
    Count occurences of pairwise combinations of motifs and compare between groups

    Args:
        counts (pd.DataFrame): Pandas dataframe containing the motif count matrix.
            Rows should be sequences and columns should be motifs.
        seqs (pd.DataFrame): Pandas dataframe containing sequences
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in `seqs` containing group IDs
        min_group_freq (int): Limit to combinations with this number of occurences in
            at least one group.
        min_group_prop (float): Limit to combinations with this proportion of occurences
            in at least one group.

    Returns:
        res (pd.DataFrame): Pandas dataframe containing FDR-corrected significance
            testing results for the occurrence of pairwise combinations between groups
    """

    print("Listing motif combinations")

    # Get the complete set of motifs present in each sequence
    motif_combinations = counts.apply(
        lambda row: set(counts.columns[np.where(row > 0)[0]]), axis=1
    ).reset_index()
    motif_combinations.columns = ["SeqID", "motifs"]
    motif_combinations = motif_combinations.merge(
        seqs[["SeqID", group_col]], on="SeqID"
    )

    # Get pairwise combinations present in each sequence
    motif_combinations["combination"] = motif_combinations.motifs.apply(
        lambda x: list(itertools.combinations(x, 2))
    )
    motif_combinations = motif_combinations[["SeqID", "combination", "Group"]].explode(
        "combination"
    )

    print("Making count matrix")
    # Count the number of sequences in which each motif combination is present
    cts = (
        motif_combinations[["combination", "Group"]]
        .value_counts()
        .reset_index(name="count")
    )

    print("Filtering")
    # Drop rare combinations
    if min_group_freq is not None:
        comb_max = cts.groupby("combination")["count"].max()
        sel_comb = cts.combination[
            cts.combination.isin(comb_max[comb_max > min_group_freq].index)
        ].tolist()
        print(f"Selected {len(sel_comb)} combinations")
        motif_combinations = motif_combinations[
            motif_combinations.combination.isin(sel_comb)
        ]
        cts = cts[cts.combination.isin(sel_comb)]

    elif min_group_prop is not None:
        cts["group_total"] = seqs.Group.value_counts()[cts.Group].tolist()
        cts["group_prop"] = cts["count"] / cts.group_total
        # print(cts)
        sel_comb = set(cts.combination[cts.group_prop > min_group_prop])
        print(f"Selected {len(sel_comb)} combinations")
        motif_combinations = motif_combinations[
            motif_combinations.combination.isin(sel_comb)
        ]
        cts = cts[cts.combination.isin(sel_comb)]

    print("Significance testing")
    df = seqs[["SeqID", "Group"]].copy()
    res = pd.DataFrame()

    for comb in cts.combination:
        seqs_with_comb = motif_combinations.SeqID[
            motif_combinations.combination == comb
        ].tolist()
        df["has_comb"] = df.SeqID.isin(seqs_with_comb)
        curr_res = groupwise_fishers(
            df,
            reference_group=reference_group,
            val_col="has_comb",
            reference_val=None,
            group_col=group_col,
        ).reset_index()
        curr_res["combination"] = [comb] * len(curr_res)
        res = pd.concat([res, curr_res])

    # FDR correction
    res["padj"] = fdrcorrection(res.pval)[1]

    return res.reset_index(drop=True)
