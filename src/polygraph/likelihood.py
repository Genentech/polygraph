import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def load_hyenadna(hyena_path, ckpt_dir=".", model="hyenadna-small-32k-seqlen"):
    """
    Loads the pretrained hyenaDNA foundation model.

    Args:
        hyena_path (str): Path to the cloned hyenaDNA repo. The repo must be cloned with
        the recurse-submodules flag. See installation instructions at
        https://github.com/HazyResearch/hyena-dna/tree/main.
        ckpt_dir (str): Path to directory in which to download the model
        model (str): Name of the foundation model to download. See
        https://github.com/HazyResearch/hyena-dna/tree/main for options.

    Returns:
        model (ConvLMHeadModel): Pretrained HyenaDNA model

    """
    # import hyenaDNA function
    sys.path.append(hyena_path)
    from src.models.sequence.long_conv_lm import ConvLMHeadModel

    # Make directory if needed
    if not os.path.exists(ckpt_dir):
        print("Making checkpoint directory")
        os.makedirs(ckpt_dir)

    # Download model if not already downloaded
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        print("Downloading model")
        config_url = (
            f"https://huggingface.co/LongSafari/{model}/resolve/main/config.json"
        )
        ckpt_url = (
            f"https://huggingface.co/LongSafari/{model}/resolve/main/weights.ckpt"
        )
        os.system(f"wget -P {ckpt_dir} {config_url}")
        os.system(f"wget -P {ckpt_dir} {ckpt_url}")

    # Load config
    print("Loading config")
    config = json.load(open(os.path.join(ckpt_dir, "config.json"), "r"))

    # Generate model
    print("Building model")
    model = ConvLMHeadModel(**config)

    # Load weights
    print("Loading weights")
    state_dict = torch.load(
        os.path.join(ckpt_dir, "weights.ckpt"), map_location=torch.device("cpu")
    )
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict["state_dict"], "model."
    )
    model_state_dict = state_dict["state_dict"]
    for key in list(model_state_dict.keys()):
        if "torchmetrics" in key:
            model_state_dict.pop(key)

    model.load_state_dict(model_state_dict)
    return model


class CharDataset(Dataset):
    def __init__(self, seqs):
        """
        A dataset class to produce sequences for hyenaDNA.

        Args:
            seqs (list): List of sequences.
        """
        self.seqs = seqs

        self.stoi = {
            "A": 7,
            "C": 8,
            "G": 9,
            "T": 10,
            "N": 11,
        }

    def __len__(self):
        return len(self.seqs)

    def encode(self, seq):
        return torch.tensor([0] + [self.stoi[base] for base in seq], dtype=torch.long)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return self.encode(seq)


def compute_likelihood(
    seqs, model, batch_size=32, num_workers=1, device="cpu", sequence_col="Sequence"
):
    """
    Function to compute log-likelihood of each sequence in the given list using the
    hyenaDNA model pretrained on the human genome.

    Args:
        seqs (pd.DataFrame): Dataframe containing DNA sequences
        model (ConvLMHead): HyenaDNA model
        batch_size (int): Batch size for inference
        num_workers (int): Number of workers for inference dataloader
        device (int, str): Device ID for inference
        sequence_col (str): Column containing sequences

    Returns
    """
    ds = CharDataset(seqs[sequence_col].tolist())
    dl = DataLoader(ds, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    model = model.to(torch.device(device))
    LL = []

    for batch in iter(dl):
        batch = batch.to(torch.device(device))
        logits = model(batch)[0][0]
        probs = F.softmax(logits, dim=2).cpu().detach()
        truth = batch[:, 1:]
        probs = probs[:, :-1, :]
        LL.extend(
            [
                np.sum([pr[pos, tru[pos]].log().item() for pos in range(pr.shape[0])])
                for pr, tru in zip(probs, truth)
            ]
        )

    return LL
