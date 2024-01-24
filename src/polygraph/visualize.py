import anndata
import numpy as np
import pandas as pd
import plotnine as p9
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage


def boxplot(data, value_col, group_col="Group", fill_col=None):
    """
    Plot boxplot
    """

    # Convert anndata to dataframe
    if isinstance(data, anndata.AnnData):
        data = data.obs

    # Fill by group if not supplied
    if fill_col is None:
        fill_col = group_col

    # Plot a single column
    return (
        p9.ggplot(data, p9.aes(x=group_col, y=value_col, fill=fill_col))
        + p9.geom_violin()
        + p9.geom_boxplot(width=0.2)
        + p9.ggtitle(f"{value_col} vs. {group_col}")
        + p9.theme_classic()
    )


def densityplot(data, value_col, group_col="Group"):
    """
    Plot density plot
    """
    return (
        p9.ggplot(data, p9.aes(x=value_col, color=group_col))
        + p9.geom_density()
        + p9.theme_classic()
    )


def plot_seqs_nmf(W, reorder=True):
    """
    Plot stacked barplot of distribution of NMF factors among sequences split by group
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
        # row_order = W.index.values
        order_dict = dict()
        for group in np.unique(groups):
            sub = W[groups == group]
            order_dict.update(dict(zip(sub.index.values, range(len(sub.index.values)))))

        # normalize
        W_norm = W.div(W.sum(axis=1), axis=0)

        # Melt dataframe
        W_norm["Group"] = groups
        W_norm = W_norm.reset_index().melt(id_vars=["SeqID", "Group"])
        W_norm["order"] = W_norm["SeqID"].map(order_dict)

        return (
            p9.ggplot(W_norm, p9.aes(x="order", y="value", fill="variable"))
            + p9.geom_col()
            + p9.facet_wrap("Group", scales="free_x")
            + p9.theme(axis_text_x=p9.element_blank())
            + p9.labels.xlab("Sequence")
            + p9.labels.ylab("Factor proportion")
            + p9.theme_classic()
        )

    else:
        # Normalize
        W_norm = W.div(W.sum(axis=1), axis=0)

        # Melt dataframe
        W_norm["Group"] = groups
        W_norm = W_norm.reset_index().melt(id_vars=["index", "Group"])

        return (
            p9.ggplot(W_norm, p9.aes(x="index", y="value", fill="variable"))
            + p9.geom_col()
            + p9.facet_wrap("Group", scales="free_x")
            + p9.theme(axis_text_x=p9.element_blank())
            + p9.labels.xlab("Sequence")
            + p9.labels.ylab("Factor proportion")
        )


def plot_factors_nmf(H, n_features=50):
    """
    Plot heatmap of contributions of features to NMF factors
    """
    H_norm = H.div(H.sum(axis=1), axis=0)
    return sns.clustermap(
        H_norm.loc[:, H_norm.max(0).sort_values().tail(n_features).index].T,
        col_cluster=False,
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
    Plot PCA embeddings of sequences
    """
    assert len(components) == 2
    df = pd.DataFrame(ad.obsm["X_pca"][:, components])
    var = np.round(ad.uns["pca"]["variance_ratio"] * 100, 2)[components]
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


def umap_plot(ad, group_col="Group", size=0.1, show_ellipse=True):
    """
    Plot UMAP embeddings of sequences
    """
    df = pd.DataFrame(ad.obsm["X_umap"])
    df.columns = ["UMAP1", "UMAP2"]
    df[group_col] = ad.obs[group_col].tolist()

    g = (
        p9.ggplot(df, p9.aes(x="UMAP1", y="UMAP2", color=group_col))
        + p9.geom_point(size=size)
        + p9.theme_classic()
    )
    if show_ellipse:
        g = g + p9.stat_ellipse()
    return g


def one_nn_frac_plot(ad, reference_group, group_col="Group", fill_col=None):
    """
    Plot a barplot showing the one-nearest neighbor fraction among reference group
    """
    totals = ad.obs[group_col].value_counts()
    nn_cts = (
        ad.obs.loc[ad.obs.one_nn_group == reference_group, group_col]
        .value_counts()
        .reset_index()
    )
    nn_cts["total"] = totals[nn_cts[group_col]].tolist()
    nn_cts["frac"] = nn_cts["count"] / nn_cts["total"]
    ref_frac = nn_cts[nn_cts[group_col] == reference_group].frac.tolist()

    g = (
        p9.ggplot(nn_cts, p9.aes(x=group_col, y="frac"))
        if fill_col is None
        else p9.ggplot(nn_cts, p9.aes(x=group_col, y="frac", fill=fill_col))
    )
    return (
        g
        + p9.geom_bar(stat="identity", position="dodge")
        + p9.theme_classic()
        + p9.theme(axis_text_x=p9.element_text(rotation=60, hjust=1))
        + p9.ylab("Fraction of sequences\nwith reference\nnearest neighbor")
        + p9.geom_hline(yintercept=ref_frac, linetype="dashed")
    )


def upset_plot(ad, group_col="Group"):
    """
    Plot UpSet plot
    """
    from upsetplot import UpSet

    groups = ad.obs[group_col].unique()

    df = pd.DataFrame()
    for group in groups:
        df[group] = ad.X[ad.obs[group_col] == group, :].sum(0) > 0
    df.columns = groups

    return UpSet(df.value_counts(), show_counts=True)
