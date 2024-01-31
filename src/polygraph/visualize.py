import anndata
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage


def boxplot(data, value_col, group_col="Group", fill_col=None):
    """
    Plot boxplot of values in each group

    Args:
        data (pd.DataFrame, anndata.AnnData): Pandas dataframe with group IDs
            and values to compare, or an AnnData object containing this
            dataframe in .obs
        value_col (str): Column containing values to plot
        group_col (str): Column containing group IDs
        fill_col (str): Column containing additional variable to split each group
    """
    if isinstance(data, pd.DataFrame):
        return (
            p9.ggplot(
                data, p9.aes(x=group_col, y=value_col, fill=fill_col or group_col)
            )
            + p9.geom_violin()
            + p9.geom_boxplot(width=0.2, outlier_size=0.1)
            + p9.ggtitle(f"{value_col} vs. {group_col}")
            + p9.theme_classic()
            + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
        )
    elif isinstance(data, anndata.AnnData):
        return boxplot(data.obs)

    else:
        raise TypeError("data must be an AnnData object or dataframe.")


def densityplot(data, value_col, group_col="Group"):
    """
    Plot density plot of values in each group

    Args:
        data (pd.DataFrame, anndata.AnnData): Pandas dataframe with group IDs
            and values to compare, or an AnnData object containing this
            dataframe in .obs
        value_col (str): Column containing values to plot
        group_col (str): Column containing group IDs
    """
    if isinstance(data, pd.DataFrame):
        return (
            p9.ggplot(data, p9.aes(x=value_col, color=group_col))
            + p9.geom_density()
            + p9.theme_classic()
        )
    elif isinstance(data, anndata.AnnData):
        return boxplot(data.obs)

    else:
        raise TypeError("data must be an AnnData object or dataframe.")


def plot_seqs_nmf(W, reorder=True):
    """
    Plot stacked barplot of the distribution of NMF factors among sequences,
    split by group

    Args:
        W (pd.DataFrame): Dataframe of shape n_seqs x (n_factors+1). The last
            column should contain group IDs.
        reorder (bool):

    """
    groups = W.iloc[:, -1].values
    W = W.iloc[:, :-1]

    if reorder:
        # cluster and get dendrogram
        Z = linkage(W, "ward")
        order = dendrogram(Z, no_plot=True)["leaves"]

        # reorder by cluster
        W = W.iloc[order, :]
        groups = groups[order]

        # get index ordering for plot
        order_dict = dict()
        for group in np.unique(groups):
            sub = W[groups == group]
            order_dict.update(dict(zip(sub.index.values, range(len(sub)))))

    # normalize
    W_norm = W.div(W.sum(axis=1), axis=0)

    # Melt dataframe
    W_norm["Group"] = groups
    W_norm = W_norm.reset_index().melt(id_vars=["SeqID", "Group"])

    if order:
        W_norm["order"] = W_norm["SeqID"].map(order_dict)

    return (
        p9.ggplot(
            W_norm,
            p9.aes(x="order" if reorder else "SeqID", y="value", fill="variable"),
        )
        + p9.geom_col()
        + p9.facet_wrap("Group", scales="free_x")
        + p9.theme_classic()
        + p9.theme(axis_text_x=p9.element_blank())
        + p9.labels.xlab("Sequence")
        + p9.labels.ylab("Factor proportion")
    )


def plot_factors_nmf(H, n_features=50, **kwargs):
    """
    Plot heatmap of contributions of features to NMF factors

    Args:
        H (pd.DataFrame): Dataframe of shape (factors, features)
        n_features (int): Number of features to cluster
        **kwargs: Additional arguments to pass to sns.clustermap
    """
    H_norm = H.div(H.sum(axis=1), axis=0)
    return sns.clustermap(
        H_norm.loc[:, H_norm.max(0).sort_values().tail(n_features).index].T,
        col_cluster=False,
        **kwargs,
    )


def pca_plot(
    ad,
    group_col="Group",
    components=[0, 1],
    size=0.1,
    show_ellipse=True,
    reference_group=None,
):
    """
    Plot PCA embeddings of sequences, colored by group.

    Args:
        ad (anndata.AnnData): AnnData object containing PCA components.
        group_col (str): Column containing group IDs.
        components (list): PCA components to plot
        size (float): Size of points
        show_ellipse (bool): Outline each group with an ellipse.
        reference_group (str): Group to use as reference. This group will
            be plotted first.
    """
    assert len(components) == 2
    df = pd.DataFrame(ad.obsm["X_pca"][:, components])
    var = np.round(ad.uns["pca"]["variance_ratio"] * 100, 2)[components]

    # Make axis labels
    col1 = f"PC{(components[0] + 1)}"
    col2 = f"PC{(components[1] + 1)}"
    xlab = col1 + f" ({str(var[0])}%)"
    ylab = col2 + f" ({str(var[1])}%)"
    df.columns = [col1, col2]
    df[group_col] = ad.obs[group_col].tolist()
    if reference_group is not None:
        df = pd.concat(
            [df[df[group_col] == reference_group], df[df[group_col] != reference_group]]
        )

    # Plot
    g = (
        p9.ggplot(df, p9.aes(x=col1, y=col2, color=group_col))
        + p9.geom_point(size=size)
        + p9.xlab(xlab)
        + p9.ylab(ylab)
        + p9.theme_classic()
    )
    if show_ellipse:
        g = g + p9.stat_ellipse()
    return g


def umap_plot(ad, group_col="Group", size=0.1, show_ellipse=True, reference_group=None):
    """
    Plot UMAP embeddings of sequences, colored by group.

    Args:
        ad (anndata.AnnData): AnnData object containing UMAP embedding.
        group_col (str): Column containing group IDs.
        size (float): Size of points
        show_ellipse (bool): Outline each group with an ellipse.
        reference_group (str): Group to use as reference. This group will
            be plotted first.
    """
    df = pd.DataFrame(ad.obsm["X_umap"])
    df.columns = ["UMAP1", "UMAP2"]
    df[group_col] = ad.obs[group_col].tolist()

    # Plot
    g = (
        p9.ggplot(df, p9.aes(x="UMAP1", y="UMAP2", color=group_col))
        + p9.geom_point(size=size)
        + p9.theme_classic()
    )
    if show_ellipse:
        g = g + p9.stat_ellipse()
    return g


def one_nn_frac_plot(ad, reference_group, group_col="Group"):
    """
    Plot a barplot showing the fraction of points in each group whose
    nearest neighbors are reference sequences.

    Args:
        ad (anndata.AnnData): AnnData object containing sequence embedding.
        reference_group (str): Group to use as reference. This group will
            be plotted first.
        group_col (str): Column in ad.obs containing group IDs.
        fill_col (str): Column containing additional variable to split each group
    """
    props = ad.uns["1NN_ref_prop_test"][["group_prop"]].reset_index()

    return (
        p9.ggplot(props, p9.aes(x=group_col, y="group_prop"))
        + p9.geom_bar(stat="identity", position="dodge")
        + p9.theme_classic()
        + p9.theme(axis_text_x=p9.element_text(rotation=90, hjust=1))
        + p9.ylab("Fraction of sequences\nwith reference\nnearest neighbor")
    )


def upset_plot(ad, group_col="Group"):
    """
    Plot UpSet plot showing the overlap between features present in
    different groups.

    Args:
        ad (anndata.AnnData): AnnData object containing sequence embedding.
        group_col (str): Column in ad.obs containing group IDs.
    """
    from upsetplot import UpSet

    groups = ad.obs[group_col].unique()

    df = pd.DataFrame()
    for group in groups:
        df[group] = ad.X[ad.obs[group_col] == group, :].sum(0) > 0
    df.columns = groups

    return UpSet(df.value_counts(), show_counts=True)
