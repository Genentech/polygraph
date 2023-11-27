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
        sequences (list): List of sequences
        batch_size (int): Batch size

    Returns:
        sequence batch generator
    """
    # Pad sequences to the same length
    padded_seqs = pad_with_Ns(sequences)

    # Yield each batch
    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        yield padded_seqs[start:end]


def load_enformer():
    """
    Load pre-trained enformer model
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
        np.array of shape (n_seqs x )
    """
    return model(sequences, return_only_embeddings=True).mean(1).cpu().detach().numpy()


def load_nucleotide_transformer(
    model="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
):
    """
    Load pre-trained nucleotide transformer model

    Args:
        model (str): Name of pretrained model to download
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForMaskedLM.from_pretrained(model)
    return model, tokenizer


def nucleotide_transformer_embed(sequences, model, tokenizer):
    """
    Embed a batch of sequences using the pre-trained nucleotide transformer model

    Args:
        sequences (list): List of sequences
        model: pre-trained nucleotide transformer model

    Returns:
        np.array of shape (n_seqs x )
    """
    tokens = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding=True)[
        "input_ids"
    ]
    torch_outs = model(
        tokens,
        attention_mask=tokens != tokenizer.pad_token_id,
        encoder_attention_mask=tokens != tokenizer.pad_token_id,
        output_hidden_states=True,
    )
    return torch_outs["hidden_states"][-1].mean(1).cpu().detach().numpy()


def sequential_embed(sequences, model, drop_last_layers, swapaxes=False):
    """
    Embed a batch of sequences using a torch.nn.Sequential model

    Args:
        sequences (list): List of sequences
        model (nn.Sequential): trained model
        drop_last_layers (int): Number of final layers to drop to get embeddings

    Returns:
        np.array of shape (n_seqs x n_features)
    """
    x = str_to_one_hot(sequences).swapaxes(1, 2)
    x = model[:-drop_last_layers](x)
    if swapaxes:
        x = x.swapaxes(1, 2)
    return x.mean(-1).cpu().squeeze().detach().numpy()


def _get_embeddings(model, sequences, drop_last_layers=1):
    """
    Get model embeddings for a batch of sequences

    Args:
        sequences (list): List of sequences
        model (nn.Sequential): trained model

    Returns:
        np.array of shape (n_seqs x n_features)
    """
    if isinstance(model, Enformer):
        return enformer_embed(sequences, model)
    elif isinstance(model, EsmForMaskedLM):
        return nucleotide_transformer_embed(sequences, model)
    elif isinstance(model, nn.Sequential):
        return sequential_embed(sequences, model, drop_last_layers)
    else:
        raise TypeError(
            "Embeddings cannot be automatically returned for this model type."
        )


def get_embeddings(seq_df, model, batch_size, drop_last_layers=1):
    """
    Get model embeddings for all sequences in a dataframe
    """
    embeddings = [
        _get_embeddings(model, x, drop_last_layers)
        for x in batch(seq_df.Sequence.tolist(), batch_size)
    ]
    return pd.DataFrame(np.vstack(embeddings), index=seq_df.index)


def cell_type_specificity(on_target, off_target):
    """
    Calculate cell type specificity from predicted or measured output
    """
    # Stack off-target values
    off_target = np.vstack(off_target)

    # Multiply the on-target values to get the same shape
    on_target = np.tile(on_target, (off_target.shape[0], 1))

    # Get the log2 fold change between on-target values and each off-target cell type
    lfc = np.log2(on_target / off_target)

    # Calculate gap statistics
    mingap = lfc.min(0)
    maxgap = lfc.max(0)
    meangap = lfc.mean(0)
    return pd.DataFrame({"mingap": mingap, "maxgap": maxgap, "meangap": meangap})


def predict(model, seq_df, batch_size, device="cpu"):
    """
    Predict using a sequence-to-function model
    """
    preds = []

    # Move model to device
    model = model.to(torch.device(device))

    # Batch the sequences
    for x in batch(seq_df.Sequence.tolist(), batch_size):
        # One-hot encode the sequence
        x = str_to_one_hot(x).swapaxes(1, 2)

        # Move to device
        x = x.to(torch.device(device))

        # Predict
        preds.append(model(x).cpu().detach().squeeze().numpy())

    return np.vstack(preds)
