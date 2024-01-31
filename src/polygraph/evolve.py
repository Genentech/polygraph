import numpy as np
import pandas as pd
import torch
import anndata
import scanpy as sc
from sklearn.metrics import pairwise_distances

from polygraph.models import get_embeddings, predict
from polygraph.sequence import ISM


def evolve(start_seq, reference_seqs, iter, model, k=None, drop_last_layers=None, batch_size=512, device="cpu", task=None, alpha=3):
    """
    Directed evolution with an additional goal to increase similarity to reference sequences.

    Args:
        start_seq (str): Start sequence
        reference_seqs (list): Reference sequences
        iter (int): Number of iterations
        model (nn.Sequential): Torch sequential model
        drop_last_layers (int): Number of terminal layers to drop from the model for model embedding.
        k (int): k-mer length for k-mer embedding.
        batch_size (int): Batch size for inference
        device (int, str): Index of device to use for inference
        task (int): Model output head. If None, average all heads.
        alpha (int): Relative weight for similarity

    Returns:
        best_seq (str): Optimized sequence
    """
    # Embed the reference sequences
    if k is not None:
        reference_embeddings = kmer_frequencies(reference_seqs, k=k, normalize=True)
    elif drop_last_layers is not None:
        reference_embeddings = get_embeddings(reference_seqs, model, batch_size=batch_size, drop_last_layers=drop_last_layers, device=device)
    else:
        raise ValueError("One of k or drop_last_layers should be provided.")
    reference_ad = anndata.AnnData(reference_embeddings)

    for i in range(1, iter + 1):
        print(f"Iter: {i}")
        if i==1:
            curr_seqs = [start_seq]

        # Embed the evolved sequences
        if k is not None:
            curr_embeddings = kmer_frequencies(curr_seqs, k=k, normalize=True)
        else:
           curr_embeddings = get_embeddings(curr_seqs, model, batch_size=batch_size, drop_last_layers=drop_last_layers, device=device) 
        curr_ad = anndata.AnnData(curr_embeddings, obs=pd.DataFrame({'Sequence':curr_seqs}))

        # Predict on evolved sequences
        curr_ad.obs['pred'] = predict(curr_seqs, model, batch_size=batch_size, device=device)

        # Combine
        ad = anndata.concat([reference_ad, curr_ad], index_unique="_", keys=["ref", "curr"])

        # PCA
        sc.pp.pca(ad, n_comps=50)

        # Get PCA embeddings for evolved and reference sequences
        reference_X = ad.obsm["X_pca"][:len(reference_ad), :]
        curr_X = ad.obsm["X_pca"][len(reference_ad):, :]

        # Get euclidean distance of each evolved sequence to its closest
        # reference sequence
        curr_ad.obs['distance'] = pairwise_distances(curr_X, reference_X, metric="euclidean").min(1)

        # Assign each sequence a total score
        curr_ad.obs['score'] = curr_ad.obs['pred'] - (alpha*curr_ad.obs['distance'])

        # Select best sequence from current iteration
        best = curr_ad.obs.sort_values('score').tail(1)
        best_seq = best.Sequence.values[0]

        # Get sequences for next round
        if i < iter:
            curr_seqs = ISM(best_seq)
        else:
            out.append(best_seq)

        print(f"Best prediction: {best.pred.values[0]} Best distance: {best.distance.values[0]} Score: {best.score.values[0]}")

