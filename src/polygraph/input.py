import os

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
resources_dir = os.path.join(script_dir, "resources")


def read_seqs(file, sep="\t", incl_ids=False):
    """
    Read sequences and group labels into a dataframe. This creates the input
    dataframe for all subsequent analyses.

    Args:
        file (str): path to a text file containing no header. If incl_ids=True,
        the first column should contain IDs and the next two columns should contain
        sequence and group label. If incl_ids=False, the first two columns should
        contain sequence and group label.
        sep (str): Column separator
        incl_ids (bool): Whether the first column corresponds to sequence IDs.

    Returns:
        df (pd.DataFrame): Pandas dataframe with columns Sequence, Group
            and a unique index.
    """
    if incl_ids:
        df = pd.read_csv(
            file,
            sep=sep,
            header=None,
            usecols=(0, 1, 2),
            names=["SeqID", "Sequence", "Group"],
            dtype="str",
        ).set_index("SeqID")
        assert len(set(df.index)) == len(df), "SeqIDs are not unique."

    else:
        from polygraph.utils import make_ids

        df = pd.read_csv(
            file,
            sep=sep,
            header=None,
            usecols=(0, 1),
            names=["Sequence", "Group"],
            dtype="str",
        )

        # Add unique IDs
        df = make_ids(df)

    return df


def read_meme_file(file):
    """
    Read a motif database in MEME format

    Args:
        file (str): path to MEME file

    Returns:
        motifs (list): List of pymemesuite.common.Motif objects
        bg (pymemesuite.common.Background): Background distribution
    """
    from pymemesuite.common import MotifFile

    # Open file
    motiffile = MotifFile(file)

    # Read motifs until file end
    motifs = []
    while True:
        motif = motiffile.read()
        if motif is None:
            break
        motifs.append(motif)

    print(f"Read {len(motifs)} motifs from file.")
    return motifs, motiffile.background


def download_jaspar(
    family="vertebrates", download_dir=os.path.join(resources_dir, "jaspar")
):
    """
    Download and read the JASPAR database of TF motifs

    Args:
        family (str): JASPAR family. one of "fungi", "insects", "nematodes",
            "plants", "urochordates", "vertebrates"
        download_dir (str): Path to directory in which to download motifs

    Returns:
        (str): Path to downloaded local file
    """
    # Create download directory
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download
    jaspar_core_prefix = (
        "https://jaspar.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_"
    )

    url = f"{jaspar_core_prefix}{family}_non-redundant_pfms_meme.txt"
    local_path = os.path.join(
        download_dir, f"JASPAR2022_CORE_{family}_non-redundant_pfms_meme.txt"
    )
    if os.path.exists(local_path):
        print(f"File already exists at {local_path}")
    else:
        os.system(f"wget --no-check-certificate -P {download_dir} {url}")
    return str(local_path)


def download_gtex_tpm(download_dir=os.path.join(resources_dir, "gtex")):
    """
    Download per-tissue TPM values from GTEX.

    Args:
        download_dir (str): Path to directory in which to download file

    Returns:
        (str): Path to downloaded local file
    """
    # Create download directory
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    url = (
        "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/"
        + "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
    )
    local_path = os.path.join(
        download_dir, "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
    )
    if os.path.exists(local_path):
        print(f"File already exists at {local_path}")
    else:
        os.system(f"wget --no-check-certificate -P {download_dir} {url}")
    return str(local_path)


def load_gtex_tpm(download_dir=os.path.join(resources_dir, "gtex")):
    """
    Load per-tissue TPM values from GTEX.

    Args:
        download_dir (str): Path to directory in which to download file

    Returns:
        (pd.DataFrame): TPM matrix.
    """
    local_path = download_gtex_tpm(download_dir)
    return pd.read_table(local_path, skiprows=2)
