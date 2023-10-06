import pandas as pd
import os
import sys
import Bio.motifs, Bio.motifs.jaspar

script_dir = os.path.dirname(os.path.abspath(__file__))
resources_dir = os.path.join(script_dir, "resources")


def read_seqs(file, sep='\t', incl_ids=False):
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
        df = pd.read_csv(file, sep=sep, header=None, usecols=(0,1,2), names=['SeqID', 'Sequence', 'Group'], dtype="str")
        assert len(set(df.SeqID)) == len(df), "SeqIDs are not unique."
    else:
        df = pd.read_csv(file, sep=sep, header=None, usecols=(0,1), names=['Sequence', 'Group'], dtype="str")

        # Add custom IDs
        df['SeqID'] = [f"seq_{i}" for i in range(len(df))]

        # Reorder columns
        df = df[['SeqID', 'Sequence', 'Group']]
        
    return df


def read_jaspar(file):
    """
    Read motifs from a JASPAR format file

    Args:
        file (str): Path to JASPAR format file

    Returns:
        List of Bio.motif objects
    
    """
    return list(Bio.motifs.jaspar.read(open(file, 'r'), format='jaspar'))


def read_pfms(dir):
    """
    Read motifs from a directory containing .pfm files

    Args:
        dir (str): Path to directory containing .pfm files

    Returns:
        motifs (list): List of Bio.motif objects
    """
    motifs=[]
    for file in os.listdir(dir):

        # get path to .pfm file
        path = os.path.join(dir, file)

        # Read
        m = Bio.motifs.parse(open(path, 'r'), fmt='pfm-four-rows', strict=True)[0]

        # Take motif name from file name
        m.name = os.path.splitext(file)[0]
        motifs.append(m)
    return motifs


def load_jaspar(family='vertebrates', download_dir=os.path.join(resources_dir, "jaspar")):
    """
    Download and read the JASPAR database of TF motifs

    Args:
        family (str): JASPAR family. one of "fungi", "insects", "nematodes", "plants", "urochordates", "vertebrates"
        download_dir (str): Path to directory in which to download motifs

    Returns:
        List of Bio.motif objects
    """
    # Create download directory
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download
    url = f"https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_{family}_non-redundant_pfms_jaspar.txt"
    local_path = os.path.join(download_dir, f"JASPAR2022_CORE_{family}_non-redundant_pfms_jaspar.txt")
    if not os.path.exists(local_path):
        os.system(f"wget -P {download_dir} {url}")

    # Read
    return read_jaspar(local_path)
    

def load_yetfasco(download_dir=os.path.join(resources_dir, "yetfasco")):
    """
    Download and read the YeTFaSCo database of yeast TF motifs

    Args:
        download_dir (str): Path to directory in which to download motifs

    Returns:
        List of Bio.motif objects
    """
    # Create download directory
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download
    url = "http://yetfasco.ccbr.utoronto.ca/1.02/Downloads/Expert_PFMs.zip"
    local_path = os.path.join(download_dir, "Expert_PFMs.zip")
    local_dir = os.path.join(download_dir, "1.02/ALIGNED_ENOLOGO_FORMAT_PFMS")
    if not os.path.exists(local_dir):
        os.system(f"wget -P {download_dir} {url}")
        os.system("unzip {} -d {}".format(local_path, download_dir))

    # Read
    return read_pfms(local_dir)
    