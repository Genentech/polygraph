import numpy as np
import pandas as pd
import torch
from enformer_pytorch import Enformer, str_to_one_hot
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, EsmForMaskedLM

from polygraph.utils import pad_with_Ns


def batch(sequences, batch_size):
    """
    Pad sequences to a constant length and split them into batches to pass to a model

    Args:
        sequences (list): List of DNA sequences
        batch_size (int): Batch size

    Returns:
        sequence batch generator
    """
    # Pad sequences to the same length
    padded_seqs = pad_with_Ns(sequences)
    batch_size = np.min([batch_size, len(sequences)])

    # Yield each batch
    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        yield padded_seqs[start:end]


def load_enformer():
    """
    Load pre-trained enformer model

    Returns:
        (Enformer): Pretrained model
    """
    return Enformer.from_pretrained(
        "EleutherAI/enformer-official-rough", target_length=-1
    )


def enformer_embed(sequences, model):
    """
    Embed a batch of sequences using pretrained or fine-tuned enformer

    Args:
        sequences (list): List of sequences
        model (Enformer): pre-trained or fine-tuned enformer model

    Returns:
        np.array of shape (n_seqs x 3072)
    """
    return model(sequences, return_only_embeddings=True).mean(1).cpu().detach().numpy()


def load_nucleotide_transformer(
    model="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
):
    """
    Load pre-trained nucleotide transformer model

    Args:
        model (str): Name of pretrained model to download

    Returns:
        model (EsmForMaskedLM): Pre-trained model
        tokenizer (): Class to convert sequences to tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForMaskedLM.from_pretrained(model)
    return model, tokenizer


def nucleotide_transformer_embed(seqs, model, tokenizer):
    """
    Embed a batch of sequences using the pre-trained nucleotide transformer model

    Args:
        sequences (list): List of sequences
        model: pre-trained nucleotide transformer model

    Returns:
        np.array of shape (n_seqs x n_features)
    """
    tokens = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True)[
        "input_ids"
    ]
    torch_outs = model(
        tokens,
        attention_mask=tokens != tokenizer.pad_token_id,
        encoder_attention_mask=tokens != tokenizer.pad_token_id,
        output_hidden_states=True,
    )
    return torch_outs["hidden_states"][-1].mean(1).cpu().detach().numpy()


def sequential_embed(seqs, model, drop_last_layers, swapaxes=False, device="cpu"):
    """
    Embed a batch of sequences using a torch.nn.Sequential model

    Args:
        seqs (list): List of sequences
        model (nn.Sequential): trained model
        drop_last_layers (int): Number of terminal layers to drop to get embeddings

    Returns:
        np.array of shape (n_seqs x n_features)
    """
    x = str_to_one_hot(seqs).swapaxes(1, 2).to(torch.device(device))
    x = model[:-drop_last_layers](x)
    if swapaxes:
        x = x.swapaxes(1, 2)
    return x.mean(-1).cpu().squeeze().detach().numpy()


def _get_embeddings(seqs, model, drop_last_layers=1, device="cpu", swapaxes=False):
    """
    Get model embeddings for a batch of sequences

    Args:
        seqs (list): List of sequences
        model (nn.Sequential): trained model
        drop_last_layers (int): Number of terminal layers to drop to get embeddings

    Returns:
        np.array of shape (n_seqs x n_features)
    """
    if isinstance(model, Enformer):
        return enformer_embed(seqs, model)
    elif isinstance(model, EsmForMaskedLM):
        return nucleotide_transformer_embed(seqs, model)
    elif isinstance(model, nn.Sequential):
        return sequential_embed(
            seqs,
            model,
            drop_last_layers=drop_last_layers,
            swapaxes=swapaxes,
            device=device,
        )
    else:
        raise TypeError(
            "Embeddings cannot be automatically returned for this model type."
        )


def get_embeddings(
    seqs, model, batch_size, drop_last_layers=1, device="cpu", swapaxes=False
):
    """
    Get model embeddings for all sequences in a dataframe

    Args:
        seqs (list, pd.DataFrame): List of sequences or dataframe
            containing sequences in the column "Sequence".
        model (nn.Sequential): trained model
        batch_size (int): Batch size for inference
        drop_last_layers (int): Number of terminal layers to drop to get embeddings
        device (str, int): ID of GPU to perform inference.
        swapaxes (bool): If true, batches will be of shape (N, 4, L).
            Otherwise, shape will be (N, L, 4).

    Returns:
        np.array of shape (n_seqs x n_features)
    """
    if isinstance(seqs, list):
        orig_device = next(model.parameters()).device
        model = model.to(torch.device(device))
        embeddings = []

        # Batch the sequences
        for x in batch(seqs, batch_size):
            embeddings.append(
                _get_embeddings(
                    x,
                    model,
                    drop_last_layers=drop_last_layers,
                    device=device,
                    swapaxes=swapaxes,
                )
            )

        model = model.to(orig_device)
        return np.vstack(embeddings)

    elif isinstance(seqs, pd.DataFrame):
        return pd.DataFrame(
            get_embeddings(
                seqs.Sequence.tolist(),
                model,
                batch_size=batch_size,
                drop_last_layers=drop_last_layers,
                device=device,
                swapaxes=swapaxes,
            ),
            index=seqs.index,
        )

    else:
        raise TypeError("seqs must be a list or dataframe.")


def cell_type_specificity(seqs, on_target_col, off_target_cols):
    """
    Calculate cell type specificity from predicted or measured output

    Args:
        seqs (pd.DataFrame): Dataframe containing sequence predictions
        on_target (str): Column containing predictions in on-target cell type
        off_target (list): Columns containing predictions in off-target cell types.

    Returns:
        (pd.DataFrame): seqs with additional columns mingap, maxgap and meangap,
            reporting 3 measures of cell type specificity for each sequence.
    """
    # Get the log2 fold change between on-target values and each off-target cell type
    lfc = np.log2(seqs[[on_target_col]].values / seqs[off_target_cols].values)

    # Calculate gap statistics
    seqs["mingap"] = lfc.min(1)
    seqs["maxgap"] = lfc.max(1)
    seqs["meangap"] = lfc.mean(1)
    return seqs


def predict(seqs, model, batch_size, device="cpu"):
    """
    Predict sequence properties using a sequence-to-function model.

    Args:
        seqs (list, pd.DataFrame): List of sequences or dataframe
            containing sequences in the column "Sequence".
        model (nn.Sequential): trained model
        batch_size (int): Batch size for inference
        device (str, int): ID of GPU to perform inference.

    Returns:
        (np.array): Array of shape (n_seqs x n_outputs)
    """
    preds = []

    # Move model to device
    model = model.to(torch.device(device))

    if isinstance(seqs, pd.DataFrame):
        seqs = seqs.Sequence.tolist()

    # Batch the sequences
    for x in batch(seqs, batch_size):
        # One-hot encode the sequence
        x = str_to_one_hot(x).swapaxes(1, 2)

        # Move to device
        x = x.to(torch.device(device))

        # Predict
        preds.append(model(x).cpu().detach().numpy())  # N, T

    return np.vstack(preds).squeeze()


def ism_score(model, seqs, batch_size, device="cpu", task=None):
    """
    Get base-level importance scores for given sequence(s) using ISM

    Args:
        seqs (list, pd.DataFrame): List of sequences or dataframe
            containing sequences in the column "Sequence".
        model (nn.Sequential): trained model
        batch_size (int): Batch size for inference
        device (str, int): ID of GPU to perform inference.

    Returns:
        (pd.DataFrame): DataFrame of shape (n_seqs x n_outputs)
    """
    from polygraph.sequence import ISM
    from polygraph.utils import check_equal_lens

    assert check_equal_lens(seqs)

    # Predictions on original sequences
    preds = predict(seqs=seqs, model=model, batch_size=batch_size, device=device)
    assert preds.ndim < 3

    # Select relevant task/cell type, or average predictions
    if task is None:
        if preds.ndim == 2:
            preds = preds.mean(1, keepdims=True)
    else:
        preds = preds[:, task]

    # Mutate sequences
    ism = ISM(seqs)  # N x L x 4

    # Make predictions on mutated sequences
    ism_preds = predict(seqs=ism, model=model, batch_size=batch_size, device=device)

    # Select relevant task/cell type, or average predictions
    if task is None:
        if ism_preds.ndim == 2:
            ism_preds = ism_preds.mean(1)
    else:
        ism_preds = ism_preds[:, task]

    # Reshape predictions : N, L, 4
    ism_preds = ism_preds.reshape(len(seqs), len(ism) // (len(seqs) * 4), 4)
    ism_preds = ism_preds.max(-1)

    # Compute base-level importance score
    preds = np.abs(ism_preds - np.expand_dims(preds, 1))
    return preds


def robustness(model, seqs, batch_size, device="cpu", task=None, aggfunc="mean"):
    """
    Get robustness scores for given sequence(s) using ISM

    Args:
        seqs (list, pd.DataFrame): List of sequences or dataframe
            containing sequences in the column "Sequence".
        model (nn.Sequential): trained model
        batch_size (int): Batch size for inference
        device (str, int): ID of GPU to perform inference.
        aggfunc (str): Either 'mean' or 'max'. Determines how to aggregate the
            effect of all possible single-base mutations.

    Returns:
        (pd.DataFrame): DataFrame of shape (n_seqs x n_outputs)
    """
    from polygraph.sequence import ISM
    from polygraph.utils import check_equal_lens

    assert check_equal_lens(seqs)

    # Predictions on original sequences
    preds = predict(seqs=seqs, model=model, batch_size=batch_size, device=device)
    assert preds.ndim < 3

    # Select relevant task/cell type, or average predictions
    if task is None:
        if preds.ndim == 2:
            preds = preds.mean(1, keepdims=True)
    else:
        preds = preds[:, [task]]

    # Mutate sequences
    ism = ISM(seqs, drop_ref=True)  # N x L x 3

    # Make predictions on mutated sequences
    ism_preds = predict(seqs=ism, model=model, batch_size=batch_size, device=device)

    # Select relevant task/cell type, or average predictions
    if task is None:
        if ism_preds.ndim == 2:
            ism_preds = ism_preds.mean(1)
    else:
        ism_preds = ism_preds[:, task]

    # Reshape predictions : N, Lx3
    ism_preds = ism_preds.reshape(len(seqs), len(ism) // len(seqs))

    # Compare mutated sequences to originals
    deltas = np.abs((ism_preds / preds) - 1)

    # Aggregate over all possible mutations
    if aggfunc == "mean":
        return np.mean(deltas, 1)
    elif aggfunc == "max":
        return np.max(deltas, 1)
