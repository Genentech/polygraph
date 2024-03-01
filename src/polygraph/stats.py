import anndata
import numpy as np
import pandas as pd
from scikit_posthocs import posthoc_dunn
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection


def groupwise_fishers(
    data, reference_group, val_col, reference_val=None, group_col="Group"
):
    """
    Perform Fisher's exact test for proportions between each non-reference group
        and the reference group.

    Args:
        data (pd.DataFrame, anndata.AnnData): Pandas dataframe with group IDs
            and values to compare, or an AnnData object containing this
            dataframe in .obs
        val_col (str): Name of column with values to compare
        reference_group (str): ID of group to use as reference
        reference_val (str): A specific value whose proportion is to be compared
            between groups
        group_col (str): Name of column containing group IDs

    Returns:
        (pd.DataFrame): Dataframe containing group proportions and FDR-corrected
            p-values for each group.
    """
    if isinstance(data, pd.DataFrame):
        # List groups
        groups = data[group_col].unique()

        # Make all-group contingency table
        cont = pd.DataFrame(
            np.zeros((len(groups), 2)),
            index=groups,
            columns=[True, False],
        )
        if reference_val is None:
            reference_val = True
        for group in groups:
            cont.loc[group, True] = sum(
                (data[group_col] == group) & (data[val_col] == reference_val)
            )
            cont.loc[group, False] = sum(
                (data[group_col] == group) & (data[val_col] != reference_val)
            )
        cont = cont.astype(np.int64)

        # Compute proportion in the reference group
        ref_prop = cont.loc[reference_group][True] / cont.loc[reference_group].sum()

        # Perform Fisher's exact test for each group vs. the reference group
        res = pd.DataFrame()

        # List nonreference groups
        nonreference_groups = groups[groups != reference_group]
        for group in nonreference_groups:
            # Subset contingency table
            group_cont = cont.loc[[group, reference_group], :]
            group_prop = group_cont.loc[group][True] / group_cont.loc[group].sum()
            group_res = pd.DataFrame(
                {
                    group_col: [group],
                    "group_prop": [group_prop],
                    "ref_prop": [ref_prop],
                    "log2FC": [np.log2(group_prop / ref_prop)],
                    "pval": [
                        fisher_exact(group_cont.values, alternative="two-sided").pvalue
                    ],
                }
            )
            res = pd.concat([res, group_res])

        # Perform FDR correction
        res["padj"] = fdrcorrection(res.pval)[1]
        return res.set_index(group_col)

    elif isinstance(data, anndata.AnnData):
        return groupwise_fishers(data.obs)

    else:
        raise TypeError("data must be an AnnData object or a DataFrame.")


def groupwise_mann_whitney(data, val_col, reference_group, group_col="Group"):
    """
    Compare the mean values between each non-reference group and the
        reference group using the Mann-Whitney U test.

    Args:
        data (pd.DataFrame, anndata.AnnData): Pandas dataframe containing group IDs
            and values to compare, or an AnnData object containing this dataframe
            in .obs
        val_col (str): Name of column with values to compare
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column containing group IDs

    Returns:
        (pd.DataFrame): Dataframe containing FDR-corrected p-values for each group.
    """
    if isinstance(data, pd.DataFrame):
        # List groups
        groups = data[group_col].unique()

        # List nonreference groups
        nonreference_groups = groups[groups != reference_group]

        # Get reference values
        ref_data = data.loc[data[group_col] == reference_group, val_col]
        ref_mean = ref_data.mean()

        # Perform tests
        res = pd.DataFrame()
        for group in nonreference_groups:
            # Subset contingency table
            group_data = data.loc[data[group_col] == group, val_col]
            group_mean = group_data.mean()
            group_res = pd.DataFrame(
                {
                    group_col: [group],
                    "group_mean": [group_mean],
                    "ref_mean": [ref_mean],
                    "log2FC": [np.log2(group_mean / ref_mean)],
                    "pval": [
                        mannwhitneyu(
                            group_data, ref_data, alternative="two-sided"
                        ).pvalue
                    ],
                }
            )
            res = pd.concat([res, group_res])

        if len(res) > 1:
            # FDR corrections
            res["padj"] = fdrcorrection(res.pval)[1]

        return res

    elif isinstance(data, anndata.AnnData):
        return groupwise_mann_whitney(data.obs)

    else:
        raise TypeError("data must be an AnnData object or a DataFrame.")


def kruskal_dunn(data, val_col, group_col="Group"):
    """
    Compare the mean values between all groups using the Kruskal-Wallis
        test followed by Dunn's post-hoc test

    Args:
        data (pd.DataFrame, anndata.AnnData): Pandas dataframe with group IDs
            and values to compare, or an AnnData object containing this dataframe
            in .obs
        val_col (str): Name of column with values to compare
        group_col (str): Name of column containing group IDs

    Returns:
        (dict): Dictionary containing p-values for both Kruskal-Wallis and Dunn's test.
    """
    if isinstance(data, pd.DataFrame):
        # List groups
        groups = data[group_col].unique()

        # Kruskal-Wallis test
        pval = kruskal(
            *[data.loc[data[group_col] == group, val_col] for group in groups]
        ).pvalue

        # Dunn's test
        padj = posthoc_dunn(
            data, val_col=val_col, group_col=group_col, p_adjust="fdr_bh"
        )

        return {"Kruskal": pval, "Dunn": padj}

    elif isinstance(data, anndata.AnnData):
        return kruskal_dunn(data.obs)

    else:
        raise TypeError("data must be an AnnData object or a DataFrame.")
