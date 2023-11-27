import anndata
import pandas as pd
from scikit_posthocs import posthoc_dunn
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection


def groupwise_fishers(
    data, reference_group, val_col, reference_val=None, group_col="Group"
):
    """
    Pairwise fisher's exact test for proportions between each non-reference group
        and the reference group

    Args:
        data (pd.DataFrame, anndata.AnnData): Pandas dataframe with group IDs
            and values to compare, or anndata object containing this dataframe in .obs
        val_col (str): Name of column with values to compare
        reference_group (str): ID of group to use as reference
        reference_val (str): value whose proportion is to be compared
        group_col (str): Name of column containing group IDs

    Returns:
        pd.DataFrame with results
    """
    if isinstance(data, anndata.AnnData):
        data = data.obs.copy()

    # List groups
    groups = data[group_col].unique()

    # List nonreference groups
    nonreference_groups = groups[groups != reference_group]

    # Select group and value columns
    cont = data[[group_col, val_col]]

    # If desired, binarize the values
    if reference_val is not None:
        cont[val_col] = cont[val_col] == reference_val

    # Make all-group contingency table
    cont = cont.value_counts().unstack().fillna(0).astype(int)

    # Perform tests
    res = pd.DataFrame()
    ref_prop = cont.loc[reference_group, :].iloc[1] / cont.loc[reference_group, :].sum()
    for group in nonreference_groups:
        # Subset contingency table
        group_cont = cont.loc[[group, reference_group], :].values
        group_prop = group_cont[0, 1] / group_cont[0, :].sum()
        group_res = pd.DataFrame(
            {
                group_col: [group],
                "group_prop": [group_prop],
                "ref_prop": [ref_prop],
                "pval": [fisher_exact(group_cont, alternative="two-sided").pvalue],
            }
        )
        res = pd.concat([res, group_res])

    # FDR corrections
    res["padj"] = fdrcorrection(res.pval)[1]

    return res.set_index(group_col)


def groupwise_mann_whitney(data, val_col, reference_group, group_col="Group"):
    """
    Comparing the mean values between each non-reference group and the
        reference group using the Mann-Whitney U test.

    Args:
        data (pd.DataFrame, anndata.AnnData): Pandas dataframe with group IDs
            and values to compare, or anndata object containing this dataframe in .obs
        val_col (str): Name of column with values to compare
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column containing group IDs

    Returns:
        pd.DataFrame with results
    """
    if isinstance(data, anndata.AnnData):
        data = data.obs.copy()

    # List groups
    groups = data[group_col].unique()

    # List nonreference groups
    nonreference_groups = groups[groups != reference_group]

    # Perform tests
    res = pd.DataFrame(
        {
            "group": nonreference_groups,
            "pval": [
                mannwhitneyu(
                    data.loc[data[group_col] == group, val_col],
                    data.loc[data[group_col] == reference_group, val_col],
                    alternative="two-sided",
                ).pvalue
                for group in nonreference_groups
            ],
        }
    )

    if len(res) > 1:
        # FDR corrections
        res["padj"] = fdrcorrection(res.pval)[1]

    return res


def kruskal_dunn(data, val_col, group_col="Group"):
    """
    Comparing the mean values between all groups using the Kruskal-Wallis
        test followed by Dunn's post-hoc test

    Args:
        data (pd.DataFrame, anndata.AnnData): Pandas dataframe with group IDs
            and values to compare, or anndata object containing this dataframe in .obs
        val_col (str): Name of column with values to compare
        group_col (str): Name of column containing group IDs

    Returns:
        pd.DataFrame with results
    """
    if isinstance(data, anndata.AnnData):
        data = data.obs.copy()

    # List groups
    groups = data[group_col].unique()

    # Kruskal-Wallis test
    pval = kruskal(
        *[data.loc[data[group_col] == group, val_col] for group in groups]
    ).pvalue

    # Dunn's test
    padj = posthoc_dunn(
        data, val_col=val_col, group_col=group_col, p_adjust="bonferroni"
    )

    return {"Kruskal": pval, "Dunn": padj}
