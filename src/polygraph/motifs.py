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
        pd.DataFrame containing columns 'MotifID', 'SeqID', 'start', 'end', 'strand'.
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


def get_motif_pairs(sites):
    """
    List the pairs of motifs present in each sequence.

    Args:
        sites (pd.DataFrame): Pandas dataframe containing FIMO output.

    Returns:
        pairs (pd.DataFrame): Dataframe containing all motif pairs in each
        sequence with their orientation and distance.
    """
    # Get midpoint of motif
    sites["mid"] = sites["start"] + ((sites["end"] - sites["start"]) / 2)

    # List the motif, strand and position for each site
    df = sites[["SeqID"]].copy()
    df["data"] = sites[["MotifID", "strand", "mid"]].apply(
        lambda row: row.tolist(), axis=1
    )

    # Get all pairs of sites present in each sequence
    pairs = pd.DataFrame(
        df.groupby("SeqID")["data"].apply(lambda x: list(itertools.combinations(x, 2))),
        columns=["data"],
    )
    pairs = pairs.explode("data")

    # Get the orientation and distance for each pair of motifs
    pairs = pd.DataFrame(
        pairs.data.apply(lambda x: list(zip(*x))).tolist(),
        index=pairs.index,
        columns=["MotifID", "strand", "pos"],
    )
    pairs["orientation"] = pairs.strand.apply(
        lambda x: "same" if len(set(x)) == 1 else "opposite"
    )
    pairs["distance"] = pairs.pos.apply(lambda x: np.abs(x[1] - x[0]))
    return pairs[["MotifID", "orientation", "distance"]]


def _filter_motif_pairs(
    motif_pairs, seqs, group_col="Group", min_group_freq=0, min_group_prop=0
):
    # Count occurrence of motif pairs in each group
    cts = motif_pairs[["MotifID", group_col]].reset_index().drop_duplicates()
    cts = cts[["MotifID", "Group"]].value_counts().reset_index(name="count")

    # Filter rare pairs by frequency
    if min_group_freq > 0:
        pair_max = cts.groupby("MotifID")["count"].max()
        sel_pairs = set(pair_max[pair_max > min_group_freq].index)
        print(f"Selected {len(sel_pairs)} pairs based on maximum in-group frequency")
        motif_pairs = motif_pairs[motif_pairs.MotifID.isin(sel_pairs)]

    # Filter rare pairs by proportion
    if min_group_prop > 0:
        cts["group_total"] = seqs.Group.value_counts()[cts.Group].tolist()
        cts["group_prop"] = cts["count"] / cts.group_total
        sel_pairs = set(cts.MotifID[cts.group_prop > min_group_prop])
        print(f"Selected {len(sel_pairs)} pairs based on maximum in-group proportion")
        motif_pairs = motif_pairs[motif_pairs.MotifID.isin(sel_pairs)]

    return motif_pairs.copy()


def motif_pair_differential_abundance(
    motif_pairs,
    seqs,
    reference_group,
    group_col="Group",
    min_group_freq=0,
    min_group_prop=0,
):
    """
    Compare the rate of occurence of pairwise combinations of motifs between groups

    Args:
        motif_pairs (pd.DataFrame): Pandas dataframe containing the ouptut of
            get_motif_pairs.
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
    res = pd.DataFrame()
    motif_pairs = motif_pairs.merge(
        seqs[[group_col]], left_index=True, right_index=True
    )

    if (min_group_freq > 0) or (min_group_prop > 0):
        motif_pairs = _filter_motif_pairs(
            motif_pairs,
            seqs,
            group_col=group_col,
            min_group_freq=min_group_freq,
            min_group_prop=min_group_prop,
        )

    df = seqs[[group_col]].copy()
    for pair in motif_pairs.MotifID.unique():
        seqs_with_pair = motif_pairs[motif_pairs.MotifID == pair].index
        df["has_pair"] = df.index.isin(seqs_with_pair)
        curr_res = groupwise_fishers(
            df,
            reference_group=reference_group,
            val_col="has_pair",
            reference_val=None,
            group_col=group_col,
        ).reset_index()
        curr_res["MotifID"] = [pair] * len(curr_res)
        res = pd.concat([res, curr_res])

    # FDR correction
    res["padj"] = fdrcorrection(res.pval)[1]

    return res.reset_index(drop=True)


def motif_pair_differential_orientation(
    motif_pairs,
    seqs,
    reference_group,
    group_col="Group",
    min_group_freq=0,
    min_group_prop=0,
):
    """
    Compare the mutual orientation of all motif pairs between groups.

    Args:
        motif_pairs (pd.DataFrame): Pandas dataframe containing the ouptut of
            get_motif_pairs.
        seqs (pd.DataFrame): Pandas dataframe containing sequences
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in `seqs` containing group IDs
        min_group_freq (int): Limit to combinations with this number of occurences in
            at least one group.
        min_group_prop (float): Limit to combinations with this proportion of occurences
            in at least one group.

    Returns:
        res (pd.DataFrame): Pandas dataframe containing FDR-corrected significance
            testing results for the mutual orientation of pairwise combinations
            between groups

    """
    res = pd.DataFrame()
    motif_pairs = motif_pairs.merge(
        seqs[[group_col]], left_index=True, right_index=True
    )

    if (min_group_freq > 0) or (min_group_prop > 0):
        motif_pairs = _filter_motif_pairs(
            motif_pairs,
            seqs,
            group_col=group_col,
            min_group_freq=min_group_freq,
            min_group_prop=min_group_prop,
        )

    for pair in motif_pairs.MotifID.unique():
        curr_res = groupwise_fishers(
            motif_pairs,
            reference_group=reference_group,
            val_col="orientation",
            reference_val="same",
            group_col=group_col,
        ).reset_index()
        curr_res["MotifID"] = [pair] * len(curr_res)
        res = pd.concat([res, curr_res])

    # FDR correction
    res["padj"] = fdrcorrection(res.pval)[1]

    return res.reset_index(drop=True)


def motif_pair_differential_distance(
    motif_pairs,
    seqs,
    reference_group,
    group_col="Group",
    min_group_freq=0,
    min_group_prop=0,
):
    """
    Compare the distance between all motif pairs across groups.

    Args:
        motif_pairs (pd.DataFrame): Pandas dataframe containing the ouptut of
            get_motif_pairs.
        seqs (pd.DataFrame): Pandas dataframe containing sequences
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in `seqs` containing group IDs
        min_group_freq (int): Limit to combinations with this number of occurences in
            at least one group.
        min_group_prop (float): Limit to combinations with this proportion of occurences
            in at least one group.

    Returns:
        res (pd.DataFrame): Pandas dataframe containing FDR-corrected significance
            testing results for the distance between paired motifs, between groups
    """
    res = pd.DataFrame()
    motif_pairs = motif_pairs.merge(
        seqs[[group_col]], left_index=True, right_index=True
    )

    if (min_group_freq > 0) or (min_group_prop > 0):
        motif_pairs = _filter_motif_pairs(
            motif_pairs,
            seqs,
            group_col=group_col,
            min_group_freq=min_group_freq,
            min_group_prop=min_group_prop,
        )

    for pair in motif_pairs.MotifID.unique():
        curr_res = groupwise_mann_whitney(
            motif_pairs,
            reference_group=reference_group,
            val_col="distance",
            group_col=group_col,
        ).reset_index()
        curr_res["MotifID"] = [pair] * len(curr_res)
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
