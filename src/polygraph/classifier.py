from collections import defaultdict

import pandas as pd
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def groupwise_svm(
    ad,
    reference_group,
    group_col="Group",
    cv=5,
    is_kernel=True,
    max_iter=1000,
    use_pca=False,
):
    """
    Train an SVM to distinguish between each non-reference group and the reference group

    Args:
        ad (anndata.AnnData): Anndata object containing sequence embeddings
            of shape (n_seqs x n_vars)
        reference_group (str): ID of group to use as reference
        group_col (str): Name of column in .obs containing group ID
        cv (int): Number of cross-validation folds
        is_kernel (bool): Whether ad.X is a symmetric kernel matrix
        max_iter (int): Maximum number of iterations for SVM
        use_pca (bool): Whether to use PCA distances

    Returns:
        ad (anndata.AnnData): Modified anndata object containing each
            sequence's predicted label in .obs, as well as SVM
            performance metrics in ad.uns["svm_performance"]
    """

    # List groups
    groups = ad.obs[group_col].unique()

    # List nonreference groups
    nonreference_groups = groups[groups != reference_group]

    # Get indices of reference sequences
    is_ref = ad.obs[group_col] == reference_group

    # Dictionary to store performance metrics
    perf = defaultdict(list)

    # Train SVM per group
    for group in nonreference_groups:
        # Select sequences for comparison
        is_group = ad.obs[group_col] == group
        sel = (is_ref | is_group).values

        # Get train and test matrices
        if use_pca:
            Xtrain = ad[sel, :].obsm["X_pca"]
        else:
            Xtrain = ad[sel, :].X
            if is_kernel:
                Xtrain = Xtrain[:, sel]

        # Get group labels
        Ytrain = pd.Categorical(
            ad[sel, :].obs[group_col], categories=[group, reference_group]
        ).codes

        # Train SVM
        svm = LinearSVC(C=2, max_iter=max_iter)
        clf = CalibratedClassifierCV(svm, cv=cv).fit(Xtrain, Ytrain)

        # Get predictions
        preds = clf.predict(Xtrain)
        ad.obs.loc[sel, f"{group}_SVM_predicted_reference"] = preds

        # Get metrics
        acc = clf.score(Xtrain, Ytrain)
        auc = metrics.roc_auc_score(Ytrain, preds)
        perf[group_col].append(group)
        perf["Accuracy"].append(acc)
        perf["AUROC"].append(auc)

    ad.uns["svm_performance"] = pd.DataFrame(perf).set_index(group_col)
    return ad
