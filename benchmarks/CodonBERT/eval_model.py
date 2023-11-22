"""Evaluate model's attention mechanism. Config settings are saved in ./config.yaml."""
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertForSequenceClassification

from load_utils import load_fasta_seqs, load_npy_seqs

sys.path.append("..")
from utils.tokenizer import get_tokenizer


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg):
    tokenizer = get_tokenizer()
    ####### Loading seqs
    seqs = load_fasta_seqs(cfg.eval.data_path)
    ####### Loading model
    model = BertForSequenceClassification.from_pretrained(cfg.eval.model_dir)
    model.eval()
    ####### prediction
    layers = np.zeros([12, 12, len(seqs[0]) + 2, len(seqs[0]) + 2])
    for i_seq, seq in enumerate(seqs):
        input_ids = tokenizer.encode(" ".join(seq)).ids
        input_ids = torch.tensor([input_ids], dtype=torch.int64)  # batch_size = 1

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
            pred, attentions = outputs[:2]

            for i in range(12):
                head_att = torch.squeeze(attentions[i]).numpy()
                layers[i, :, :, :] += head_att

    fig, axs = plt.subplots(12, 12, sharey=True, sharex=True, figsize=(60, 60))
    for i_layer in range(12):
        for i_head in range(12):
            axs[i_layer][i_head].imshow(
                layers[i_layer, i_head, :, :] / len(seqs),
                cmap="hot",
                interpolation="nearest",
                origin="lower",
            )
            axs[i_layer][i_head].set_xlabel("layer:%d, head:%d" % (i_layer, i_head))

    fig.savefig("attention.pdf")


if __name__ == "__main__":
    main()
