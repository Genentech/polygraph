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
        df (pd.DataFrame): Pandas dataframe with columns SeqID, Sequence, Group
    """
    if incl_ids:
        df = pd.read_csv(
            file,
            sep=sep,
            header=None,
            usecols=(0, 1, 2),
            names=["SeqID", "Sequence", "Group"],
            dtype="str",
        )
        assert len(set(df.SeqID)) == len(df), "SeqIDs are not unique."
    else:
        df = pd.read_csv(
            file,
            sep=sep,
            header=None,
            usecols=(0, 1),
            names=["Sequence", "Group"],
            dtype="str",
        )

        # Add custom IDs
        df["SeqID"] = [f"seq_{i}" for i in range(len(df))]

        # Reorder columns
        df = df[["SeqID", "Sequence", "Group"]]

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
        motifs (list): List of pymemesuite.common.Motif objects
        bg (pymemesuite.common.Background): Background distribution
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
    assert not os.path.exists(local_path), f"File already exists at {local_path}"
    os.system(f"wget --no-check-certificate -P {download_dir} {url}")

    return str(local_path)
