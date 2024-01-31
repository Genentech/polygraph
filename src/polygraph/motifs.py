import itertools

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from statsmodels.stats.multitest import fdrcorrection

from polygraph.stats import groupwise_fishers, groupwise_mann_whitney


def scan(seqs, meme_file, group_col="Group", pthresh=1e-3, rc=True):
    """
    Scan a DNA sequence using motifs from a MEME file.

    Args:
        seqs (str): Dataframe containing DNA sequences
        meme_file (str): Path to MEME file
        group_col (str): Column containing group IDs
        pthresh (float): p-value cutoff for binding sites
        rc (bool): Whether to scan the sequence reverse complement as well

    Returns:
        pd.DataFrame containing columns 'Matrix_id', 'seq_id', 'start', 'end', 'strand'.
    """
    from collections import defaultdict

    from pymemesuite.common import Sequence
    from pymemesuite.fimo import FIMO

    from polygraph.input import read_meme_file

    # Load motifs
    motifs, bg = read_meme_file(meme_file)

    # Format sequences
    sequences = [
        Sequence(seq, name=id.encode())
        for seq, id in zip(seqs["Sequence"].tolist(), seqs.index.tolist())
    ]

    # Setup FIMO
    fimo = FIMO(both_strands=rc, threshold=pthresh)

    # Empty dictionary for output
    out = defaultdict(list)

    # Scan
    for motif in motifs:
        match = fimo.score_motif(motif, sequences, bg).matched_elements
        for m in match:
            out["MotifID"].append(motif.name.decode())
            out["SeqID"].append(m.source.accession.decode())
            out["start"].append(m.start)
            out["end"].append(m.stop)
            out["strand"].append(m.strand)

    return pd.DataFrame(out).merge(seqs[[group_col]], left_on="SeqID", right_index=True)


def motif_frequencies(sites, normalize=False, seqs=None):
    """
    Count frequency of occurrence of motifs in a list of sequences

    Args:
        sites (list): Output of `scan` function
        normalize (bool): Whether to normalize the resulting count matrix
            to correct for sequence length
        seqs (pd.DataFrame): Pandas dataframe containing DNA sequences.
            Needed if normalize=True.

    Returns:
        cts (pd.DataFrame): Count matrix with rows = sequences and columns = motifs
    """
    motifs = sites.MotifID.unique()
    cts = np.zeros((len(seqs), len(motifs)))
    cts = sites[["MotifID", "SeqID"]].value_counts().reset_index(name="count")
    cts = cts.pivot_table(index="SeqID", columns="MotifID", values="count")
    cts = cts.merge(seqs[[]], left_index=True, right_index=True, how="right").fillna(0)

    if normalize:
        assert seqs is not None, "seqs must be provided for normalization"
        seq_lens = seqs["Sequence"].apply(len)
        cts = cts.divide(seq_lens.tolist(), axis=0)
    return cts


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
    motif_combinations = pd.DataFrame(
        counts.apply(lambda row: set(counts.columns[np.where(row > 0)[0]]), axis=1)
    )
    motif_combinations.columns = ["motifs"]
    motif_combinations = motif_combinations.merge(
        seqs[[group_col]], left_index=True, right_index=True
    )

    # Get pairwise combinations present in each sequence
    motif_combinations["combination"] = motif_combinations.motifs.apply(
        lambda x: list(itertools.combinations(x, 2))
    )
    motif_combinations = motif_combinations[["combination", group_col]].explode(
        "combination"
    )

    print("Making count matrix")
    # Count the number of sequences in which each motif combination is present
    cts = (
        motif_combinations[["combination", group_col]]
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
    df = seqs[[group_col]].copy()
    res = pd.DataFrame()

    for comb in cts.combination.unique():
        seqs_with_comb = motif_combinations[
            motif_combinations.combination == comb
        ].index.tolist()
        df["has_comb"] = df.index.isin(seqs_with_comb)
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


def score_sites(sites, seqs, scores):
    """
    Calculate the average score of each motif site given base-level importance scores.

    Args:
        sites (pd.DataFrame): Dataframe containing site positions
        seqs (pd.DataFrame): Dataframe containing sequences
        scores (np.array): Numpy array of shape (sequences x length)

    Returns
        sites (pd.DataFrame): 'sites' dataframe with an additional columns 'score'
    """
    sites["score"] = sites.apply(
        lambda row: scores[seqs.index == row.SeqID, row.start : row.end].mean()
        if row.strand == "+"
        else scores[seqs.index == row.SeqID, row.end : row.start].mean(),
        axis=1,
    )
    return sites
