import sys

import numpy as np
from Bio import SeqIO

sys.path.append("..")
from utils.tokenizer import mytok


def load_npy_seqs(data_path):
    """Load sequences from a .npy file."""
    lst_all = []
    seqs = np.load(data_path)
    for seq in seqs:
        lst_tok = mytok(seq, 3, 3)
        lst_all.append(lst_tok)
    return lst_all


def load_fasta_seqs(data_path, max_length=1024):
    """Load sequences from a .fasta file."""
    lst_all = []
    records = list(SeqIO.parse(data_path, "fasta"))
    for record in records:
        sequence = str(record.seq)
        lst_tok = mytok(sequence, 3, 3)
        if len(lst_tok) > max_length - 2:
            continue
        lst_all.append(lst_tok)
    return lst_all
