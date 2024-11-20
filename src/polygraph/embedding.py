import numpy as np
import pandas as pd
import scanpy as sc
from hotelling.stats import hotelling_t2
from scipy.stats import fisher_exact
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import fdrcorrection

from polygraph.classifier import groupwise_svm
from polygraph.stats import groupwise_fishers, groupwise_mann_whitney, kruskal_dunn


def embedding_pca(ad, **kwargs):
    """
    Perform PCA on sequence embeddings.

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings
            of shape (n_seqs x n_vars)
        **kwargs: Additional arguments to pass to sc.pp.pca

    Returns:
        ad (anndata.AnnData): Modified anndata object containing PCA results
    """
    sc.pp.pca(ad, **kwargs)
    print(
        "Fraction of total variance explained by PCA components: ",
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
    diff["log2FC"] = pd.DataFrame(ad.uns["rank_genes_groups"]["logfoldchanges"]).melt()[
        "value"
    ]

    # Add to .uns
    ad.uns["DE_test"] = diff

    return ad


def reference_1nn(ad, reference_group, group_col="Group", use_pca=False):
    """
    For each sequence, find its nearest neighbor among its own group or
    the reference group based on the sequence embeddings.

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
    res = pd.DataFrame()

    # Get reference embedding
    in_ref = ad.obs[group_col] == reference_group

    # List groups
    groups = ad.obs[group_col].unique()

    # List nonreference groups
    nonreference_groups = groups[groups != reference_group]

    for group in nonreference_groups:
        # Get group embeddings
        in_group = ad.obs[group_col] == group
        in_group_or_ref = (in_ref | in_group).tolist()
        in_group_or_ref_indices = ad.obs.index[in_group_or_ref].tolist()
        if use_pca:
            X = ad.obsm["X_pca"][in_group_or_ref, :]
        else:
            X = ad.X[in_group_or_ref, :]

        # Calculate nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(X)
        distances, indices = nbrs.kneighbors(X)
        indices = np.array(in_group_or_ref_indices)[indices[:, 1]]

        # Record whether nearest neighbor is a member of the reference group
        ad.obs.loc[in_group_or_ref, f"{group}_one_nn_idx"] = indices
        ad.obs.loc[in_group_or_ref, f"{group}_one_nn_dist"] = distances[:, 1]
        ad.obs.loc[in_group_or_ref, f"{group}_one_nn_group"] = ad.obs.loc[
            indices, group_col
        ].tolist()

        # Make contingency table
        cont = ad.obs[[group_col, f"{group}_one_nn_group"]].value_counts()
        cont = (
            cont.unstack()
            .fillna(0)
            .loc[[group, reference_group], [group, reference_group]]
            .values
        )

        # Perform tests
        group_prop = cont[0, 1] / cont[0, :].sum()
        ref_prop = cont[1, 1] / cont[1, :].sum()

        res = pd.concat(
            [
                res,
                pd.DataFrame(
                    {
                        group_col: [group],
                        "group_prop": [group_prop],
                        "ref_prop": [ref_prop],
                        "pval": [fisher_exact(cont, alternative="two-sided").pvalue],
                    }
                ),
            ]
        )

    # Save results
    res = res.set_index(group_col)
    res["padj"] = fdrcorrection(res.pval)[1]
    ad.uns["1NN_group_probs"] = res.iloc[:, :2].copy()
    ad.uns["1NN_ref_prop_test"] = res
    return ad


def all_1nn(ad, reference_group, group_col="Group", use_pca=False):
    """
    Find the group ID of each sequence's 1-nearest neighbor statistics based on the
    sequence embeddings. Compare all groups to all other groups.

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
        ad.obs[group_col].iloc[indices[:, 1].tolist()].tolist()
    )

    # Normalized count table of nearest neighbor groups
    one_nn_group_df = (
        ad.obs[[group_col, "one_nn_group"]]
        .value_counts()
        .reset_index(name="count")
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


def group_diversity(ad, n_neighbors=10, group_col="Group", use_pca=False):
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
            ad.obs.loc[in_group, "Closest reference"] = (
                ad[in_ref, :].obs_names[distances.argmin(1)].tolist()
            )
            ad.obs.loc[in_group, "Distance to closest reference"] = distances.min(1)
        else:
            # If the group is the reference group, the nearest neighbor will be
            # the sequence itself. So we find the next nearest neighbor.
            indices = []
            dists = []
            for i, row in enumerate(distances):
                # Drop the nearest neighbor
                row[i] = np.Inf
                # Take the new minimum
                dists.append(row.min())
                indices.append(row.argmin())

            # Add to .obs
            ad.obs.loc[in_group, "Closest reference"] = (
                ad[in_ref, :].obs_names[indices].tolist()
            )
            ad.obs.loc[in_group, "Distance to closest reference"] = dists

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


def distribution_shift(ad, reference_group, group_col="Group", use_pca=False):
    """
    Compare the distribution of sequences in each group to the distribution
    of reference sequences, in the embedding space. Performs Hotelling's T2
    test to compare multivariate distributions.

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings
            of shape (n_seqs x n_vars)
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in .obs containing group ID
        use_pca (bool): Whether to use PCA distances

    Returns:
        ad (anndata.AnnData): Modified anndata object containing distance to
            reference in .uns['distribution_shift'].
    """
    rows = []

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

        # Perform Hotelling's T2 test to compare to the reference
        rows.append([group] + list(hotelling_t2(group_X, ref_X)[:-1]))

    # Format dataframe
    res = pd.DataFrame(rows, columns=[group_col, "t2_stat", "fval", "pval"])
    res["padj"] = fdrcorrection(res.pval)[1]
    ad.uns["dist_shift_test"] = res.set_index(group_col)
    return ad


def geometric_sketch(ad, N, groups=None, group_col="Group", use_pca=True):
    """
    Applies geometric sketching (Hie, Brian et al. Cell Systems, Volume 8,
    Issue 6, 483 - 493.e7) to sample a subset of sequences that represent
    the diversity in the specified groups.

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings
            of shape (n_seqs x n_vars)
        N (int): Number of sequences to sample from each group
        groups (list): Names of groups from which to sample sequences. If None,
            all groups are used.
        group_col (str): Name of column in .obs containing group ID
        use_pca (bool): Whether to use PCA distances

    Returns:
        ad (anndata.AnnData): Modified anndata object containing selections in
            ad.obs['selected'].
    """
    from geosketch import gs

    ad.obs["selected"] = False
    groups = groups or ad.obs[group_col].unique()

    for group in groups:
        in_group = ad.obs[group_col] == group
        group_idx = ad.obs_names[in_group].tolist()
        if use_pca:
            group_X = ad.obsm["X_pca"][in_group, :]
        else:
            group_X = ad.X[in_group, :]

        sketch_index = gs(group_X, N=N, replace=False)
        ad.obs.loc[[group_idx[x] for x in sketch_index], "selected"] = True

    return ad


def embedding_analysis(
    matrix,
    seqs,
    reference_group,
    group_col="Group",
    n_neighbors=15,
    use_pca=False,
    max_iter=1000,
):
    """
    A single function to calculate several embedding distance-based metrics.

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
        Performance metrics for SVMs trained to classify each group from the
            reference (.uns["svm_performance"])
        Predictions by SVMs trained to classify each group from the reference
            (.obs["{group}_SVM_predicted_reference"])
    """
    from anndata import AnnData

    print("Creating AnnData object")
    ad = AnnData(matrix, obs=seqs)
    ad = ad[:, ad.X.sum(0) > 0]

    print("PCA")
    ad = embedding_pca(ad)

    print("UMAP")
    ad = embedding_umap(ad, n_neighbors=n_neighbors)

    print("Differential feature abundance")
    ad = differential_analysis(ad, reference_group, group_col)

    print("1-NN statistics")
    ad = reference_1nn(ad, reference_group, group_col, use_pca=use_pca)

    print("Within-group KNN diversity")
    ad = group_diversity(ad, n_neighbors, group_col, use_pca=use_pca)

    print("Euclidean distance to nearest reference")
    ad = dist_to_reference(ad, reference_group, group_col, use_pca=use_pca)

    print("Distribution shift")
    ad = distribution_shift(ad, reference_group, group_col, use_pca=use_pca)

    print("Train groupwise classifiers")
    ad = groupwise_svm(
        ad,
        reference_group,
        group_col=group_col,
        cv=5,
        is_kernel=False,
        max_iter=max_iter,
    )

    return ad
