"""Extract embeddings from model. Config settings are saved in ./config.yaml."""
import os
import pickle
import sys

import hydra
import torch
from transformers import BertForPreTraining

from load_utils import load_fasta_seqs

sys.path.append("..")
from utils.tokenizer import get_tokenizer


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg):
    cuda = torch.cuda.is_available()
    tokenizer = get_tokenizer()

    print("loading seqs")
    seqs = load_fasta_seqs(cfg.embed.data_path, max_length=cfg.embed.max_length)

    # https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#bertformaskedlm
    model = BertForPreTraining.from_pretrained(cfg.embed.model_dir)
    if cuda:
        model = model.cuda()
    print(f"Device: {model.device}")
    model.eval()

    data = []
    for seq in seqs:
        input_ids = tokenizer.encode(" ".join(seq)).ids
        input_ids = torch.tensor([input_ids], dtype=torch.int64)
        if cuda:
            input_ids = input_ids.cuda()

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
            _, _, hidden_states = outputs[:3]
            output_embeds = torch.squeeze(hidden_states[-1])
            data.append(output_embeds.cpu().numpy()[1:-1].tolist())

    print(f"saving to {cfg.embed.output_path}")
    with open(cfg.embed.output_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
