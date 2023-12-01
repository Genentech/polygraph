import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from polygraph.stats import groupwise_fishers, groupwise_mann_whitney, kruskal_dunn


def embedding_pca(ad, **kwargs):
    """
    Perform PCA on sequence embeddings

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings
            of shape (n_seqs x n_vars)
        **kwargs: Additional arguments to pass to sc.pp.pca

    Returns:
        ad (anndata.AnnData): Modified anndata object containing PCA results
    """
    sc.pp.pca(ad, **kwargs)
    print(
        "Fraction of variance explained: ", np.round(ad.uns["pca"]["variance_ratio"], 2)
    )
    print(
        "Fraction of total variance explained: ",
        np.sum(ad.uns["pca"]["variance_ratio"]),
    )
    return ad


def embedding_umap(ad, n_neighbors=15, **kwargs):
    """
    Perform UMAP on sequence embeddings

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings of
            shape (n_seqs x n_vars)
        **kwargs: Additional arguments to pass to sc.pp.umap

    Returns:
        ad (anndata.AnnData): Modified anndata object containing UMAP results
    """
    sc.pp.neighbors(ad, n_neighbors=n_neighbors)
    sc.tl.umap(ad, **kwargs)
    return ad


def differential_analysis(ad, reference_group, group_col="Group"):
    """
    Perform Wilcoxon rank-sum test on sequence embeddings to find features
        enriched/depleted with respect to the reference group

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings of
            shape (n_seqs x n_vars)
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in .obs containing group ID

    Returns:
        ad (anndata.AnnData): Modified anndata object containing Wilcoxon test results
    """
    # Perform test
    sc.tl.rank_genes_groups(
        ad,
        groupby=group_col,
        groups="all",
        reference=reference_group,
        rankby_abs=True,
        method="wilcoxon",
    )

    # Get the variable names
    diff = pd.DataFrame(ad.uns["rank_genes_groups"]["names"]).melt(var_name=group_col)

    # Get the statistics
    diff["score"] = pd.DataFrame(ad.uns["rank_genes_groups"]["scores"]).melt()["value"]
    diff["padj"] = pd.DataFrame(ad.uns["rank_genes_groups"]["pvals_adj"]).melt()[
        "value"
    ]

    # Add to .uns
    ad.uns["DE_test"] = diff

    return ad


def one_nn_stats(ad, reference_group, group_col="Group", use_pca=False):
    """
    Calculate the following 1-nearest neighbor statistics based on the
    sequence embeddings:
    1. Group ID of nearest neighbor
    2. Distance to nearest neighbor

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings of
            shape (n_seqs x n_vars)
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in .obs containing group ID
        use_pca (bool): Whether to use PCA distances

    Returns:
        ad (anndata.AnnData): Modified anndata object containing index, distance and
            group ID of each sequence's nearest neighbor in .obs, as well as a
            probability table of nearest neighbor groups for each group
        in .uns
    """

    # Get nearest neighbor for each sequence
    if use_pca:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(
            ad.obsm["X_pca"]
        )
        distances, indices = nbrs.kneighbors(ad.obsm["X_pca"])
    else:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(ad.X)
        distances, indices = nbrs.kneighbors(ad.X)

    # Add index, dist and group of each sequence's nearest neighbor to .obs
    ad.obs.loc[:, "one_nn_idx"] = indices[:, 1]
    ad.obs.loc[:, "one_nn_dist"] = distances[:, 1]
    ad.obs.loc[:, "one_nn_group"] = (
        ad.obs[group_col].iloc[ad.obs["one_nn_idx"].tolist()].tolist()
    )

    # Normalized count table of nearest neighbor groups
    one_nn_group_df = (
        ad.obs[[group_col, "one_nn_group"]]
        .value_counts()
        .reset_index(name='count')
        .pivot_table(index=group_col, columns="one_nn_group", values="count")
        .fillna(0)
        .astype(int)
    )
    one_nn_group_df = one_nn_group_df.div(one_nn_group_df.sum(axis=1), axis=0)
    ad.uns["1NN_group_probs"] = np.round(one_nn_group_df, 2)

    # Pairwise fisher's exact test
    ad.uns["1NN_ref_prop_test"] = groupwise_fishers(
        ad.obs,
        val_col="one_nn_group",
        reference_group=reference_group,
        reference_val=reference_group,
        group_col=group_col,
    )

    return ad


def within_group_knn_dist(ad, n_neighbors=10, group_col="Group", use_pca=False):
    """
    Calculates the mean distance of each sequence to its k nearest neighbors in the
    same group, in the embedding space. Metric of diversity

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings of
            shape (n_seqs x n_vars)
        n_neighbors (int): Number of neighbors over which to average the distance
        group_col (str): Name of column in .obs containing group ID
        use_pca (bool): Whether to use PCA distances

    Returns:
        ad (anndata.AnnData): Modified anndata object containing KNN distance in .obs
    """
    groups = ad.obs[group_col].unique()
    for group in groups:
        # Get sequences in the group
        in_group = ad.obs[group_col] == group
        if use_pca:
            group_X = ad.obsm["X_pca"][in_group, :]
        else:
            group_X = ad.X[in_group, :]

        # Find nearest neighbors within group
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(
            group_X
        )
        distances, indices = nbrs.kneighbors(group_X)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Take mean KNN dist
        distances = distances.mean(1)

        # Add KNN distance to .obs
        ad.obs.loc[in_group, "KNN Distance"] = distances

    # Mann-whitney or Kruskal-wallis test
    if len(groups) == 2:
        ad.uns["knn_dist_test"] = groupwise_mann_whitney(
            ad.obs,
            val_col="KNN Distance",
            reference_group=groups[0],
            group_col=group_col,
        )
    elif len(groups) > 2:
        ad.uns["knn_dist_test"] = kruskal_dunn(
            ad.obs, val_col="KNN Distance", group_col=group_col
        )

    return ad


def dist_to_reference(ad, reference_group, group_col="Group", use_pca=False):
    """
    Calculates the euclidean distance of each sequence to its nearest neighbor in the
    reference group, in the embedding space.

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings
            of shape (n_seqs x n_vars)
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in .obs containing group ID
        use_pca (bool): Whether to use PCA distances

    Returns:
        ad (anndata.AnnData): Modified anndata object containing distance to
            reference in .obs.
    """
    # Get reference sequences
    in_ref = ad.obs[group_col] == reference_group
    if use_pca:
        ref_X = ad.obsm["X_pca"][in_ref, :]
    else:
        ref_X = ad.X[in_ref, :]

    # List groups
    groups = ad.obs[group_col].unique()

    for group in groups:
        # Get group sequences
        in_group = ad.obs[group_col] == group
        if use_pca:
            group_X = ad.obsm["X_pca"][in_group, :]
        else:
            group_X = ad.X[in_group, :]

        # Get pairwise euclidean distance of each group sequence to each
        # reference sequence
        distances = pairwise_distances(group_X, ref_X, metric="euclidean")

        if group != reference_group:
            # Add to .obs
            ad.obs.loc[in_group, "Distance to closest reference"] = distances.min(1)
        else:
            # If the group is the reference group, the nearest neighbor will be
            # the sequence itself. So we find the next nearest neighbor.
            dlist = []
            for i, row in enumerate(distances):
                # Drop the nearest neighbor
                row = np.delete(row, i)
                # Take the new minimum
                dlist.append(row.min())

            # Add to .obs
            ad.obs.loc[in_group, "Distance to closest reference"] = dlist

    # Mann-whitney or Kruskal-wallis test
    if len(groups) == 2:
        ad.uns["ref_dist_test"] = groupwise_mann_whitney(
            ad.obs,
            val_col="Distance to closest reference",
            reference_group=reference_group,
            group_col=group_col,
        )
    elif len(groups) > 2:
        ad.uns["ref_dist_test"] = kruskal_dunn(
            ad.obs, val_col="Distance to closest reference", group_col=group_col
        )

    return ad


def embedding_analysis(
    matrix, seqs, reference_group, group_col="Group", n_neighbors=15, use_pca=False
):
    """
    A single function to calculate all embedding distance-based metrics.

    Args:
        matrix (np.array, pd.DataFrame): A probability table or embedding matrix
            of dimensions seqs x features
        seqs (pd.DataFrame): Database of sequence information including group labels
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in .obs containing group ID
        n_neighbors (int): Number of neighbors for KNN
        use_pca (bool): Whether to use PCA distances

    Returns:
        Anndata object containing the following statistics:
        Embedding PCA (.obsm['X_pca'])
        Embedding UMAP (in .obsm['X_umap'])
        Wilcoxon test comparing feature abundance in nonreference vs. reference
            groups (in .uns["DE_test"])
        Index, distance and group of each sequence's nearest neighbor (in .obs)
        Test for proportion of sequences in each group whose nearest neighbor is in the
            reference group (.uns["1NN_reference_prop_test"])
        Probability table of nearest neighbor groups for each group
            (.uns["1NN_group_probs"])
        Within-group KNN distance for each sequence (.obs['KNN Distance'])
        Test for between group difference in KNN distance (.uns["knn_dist_test"])
        Distance to the closest reference sequence for each sequence
            (.obs['Distance to closest reference'])
        Test for between group difference in Distance to the closest reference
            sequence (.uns["ref_dist_test"])
    """
    from anndata import AnnData

    print("Creating AnnData object")
    ad = AnnData(matrix)
    ad.obs = seqs
    ad = ad[:, ad.X.sum(0) > 0]

    print("PCA")
    ad = embedding_pca(ad)

    print("UMAP")
    ad = embedding_umap(ad, n_neighbors=n_neighbors)

    print("Differential feature abundance")
    ad = differential_analysis(ad, reference_group, group_col)

    print("1-NN statistics")
    ad = one_nn_stats(ad, reference_group, group_col, use_pca=use_pca)

    print("Within-group KNN diversity")
    ad = within_group_knn_dist(ad, n_neighbors, group_col, use_pca=use_pca)

    print("Euclidean distance to nearest reference")
    ad = dist_to_reference(ad, reference_group, group_col, use_pca=use_pca)
    return ad
