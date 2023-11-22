import hydra
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split

import model
import train

sys.path.append("..")
from utils.tokenizer import mytok


class CustomTextDataset(torch.utils.data.TensorDataset):
    def __init__(self, text, labels, num_labels):
        if num_labels == 1:
            self.labels = torch.tensor(labels).reshape(-1, 1).to(torch.float32)
        else:
            # classification
            self.labels = torch.tensor(labels).reshape(-1, 1).to(torch.int64)

        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text[idx], self.labels[idx]


def load_embeds(embed_file):
    """Loads the embeddings from a numpy file."""
    X = np.load(embed_file)  # [num_seq, seq_len, embed_dim]
    return X


def load_data(data_path):
    """Loads the data from a csv file."""
    df_update = pd.read_csv(data_path)
    seqs = df_update["Sequence"].values.tolist()
    y = df_update["Value"].values.tolist()
    return seqs, y


def data_split(X, y, labels):
    """Splits the data into training, validation and test sets."""
    if labels == 1:
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42
        )
    else:
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=42, stratify=y_rest
        )
    with open("./data/train_embs.npy", "wb") as f:
        np.save(f, X_train)
    with open("./train_labels.npy", "wb") as f:
        np.save(f, y_train)
    with open("./valid_embs.npy", "wb") as f:
        np.save(f, X_valid)
    with open("./valid_labels.npy", "wb") as f:
        np.save(f, y_valid)
    with open("./test_embs.npy", "wb") as f:
        np.save(f, X_test)
    with open("./test_labels.npy", "wb") as f:
        np.save(f, y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def wrap_data(X_train, y_train, X_valid, y_valid, X_test, y_test, pre_embed):
    """
    Wraps the data into tensors and creates custom datasets and data loaders.
    Args:
        X_train (List[List[int]]): The training data.
        y_train (List[int]): The training labels.
        X_valid (List[List[int]]): The validation data.
        y_valid (List[int]): The validation labels.
        X_test (List[List[int]]): The test data.
        y_test (List[int]): The test labels.
        pre_embed (bool): Flag indicating whether the data is pre-embedded.
    Returns:
        train_iter (torch.utils.data.DataLoader): Data loader for the training set.
        dev_iter (torch.utils.data.DataLoader): Data loader for the validation set.
        test_iter (torch.utils.data.DataLoader): Data loader for the test set."""
    if pre_embed:
        X_train = [torch.tensor(x, dtype=torch.float32) for x in X_train]
        X_valid = [torch.tensor(x, dtype=torch.float32) for x in X_valid]
        X_test = [torch.tensor(x, dtype=torch.float32) for x in X_test]
    else:
        X_train = [torch.tensor(x, dtype=torch.int64) for x in X_train]
        X_valid = [torch.tensor(x, dtype=torch.int64) for x in X_valid]
        X_test = [torch.tensor(x, dtype=torch.int64) for x in X_test]

    ds_train = CustomTextDataset(X_train, y_train, cfg.hyperparameters.labels)
    ds_valid = CustomTextDataset(X_valid, y_valid, cfg.hyperparameters.labels)
    ds_test = CustomTextDataset(X_test, y_test, cfg.hyperparameters.labels)

    train_iter = torch.utils.data.DataLoader(ds_train, 64, shuffle=True)
    dev_iter = torch.utils.data.DataLoader(ds_valid, 64)
    test_iter = torch.utils.data.DataLoader(ds_test, 64)

    return train_iter, dev_iter, test_iter


def build_vob(s):
    """
    Creates a vocabulary dictionary based on the step size.
    Args:
        s (int): The step size for generating k-mers.
    Returns:
        Dict[str, int]: The vocabulary dictionary with elements as keys and their corresponding indices as values.
    """
    lst_ele = list("AUGCN")
    lst_voc = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    if s == 1:
        print("Nucleotide-based model")
        lst_voc.extend(lst_ele)

    elif s == 3:
        for a1 in lst_ele:
            for a2 in lst_ele:
                for a3 in lst_ele:
                    lst_voc.extend([f"{a1}{a2}{a3}"])

    dic_voc = dict(zip(lst_voc, range(len(lst_voc))))

    return dic_voc


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg):
    ### Training
    if not cfg.predict:
        ### load data
        if cfg.input.data_path:
            seqs, y = load_data(cfg.input.data_path)
        else:
            print("Please provide data file for training...")
            exit(0)

        ### build vocabulary
        pre_embed = False
        kmer_size = 1 if cfg.settings.nuc else 3
        voc_dict = build_vob(kmer_size)
        cfg.hyperparameters.vocab_size = len(voc_dict)
        X = []
        for seq in seqs:
            X.append(mytok(seq, voc_dict, kmer_size))

        ### loading embeddings if provided
        pre_embed = False
        if cfg.input.embed_file:
            pre_embed = True
            X = load_embeds(cfg.input.embed_file)
            print("loading embedding file from %s successfully" % cfg.input.embed_file)

        ### padding
        if not pre_embed:
            max_seq_len = max([len(kmer_lst) for kmer_lst in X])
            X_padded = []
            if cfg.hyperparameters.max_len:
                for kmer_lst in X:
                    seq_len = len(kmer_lst)
                    if seq_len < cfg.hyperparameters.max_len:
                        X_padded.append(
                            kmer_lst
                            + [voc_dict["[PAD]"]]
                            * (cfg.hyperparameters.max_len - seq_len)
                        )
                    else:
                        X_padded.append(kmer_lst[: cfg.hyperparameters.max_len])
            else:
                for kmer_lst in X:
                    seq_len = len(kmer_lst)
                    X_padded.append(
                        kmer_lst + [voc_dict["[PAD]"]] * (max_seq_len - seq_len)
                    )

            X = X_padded

        X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(
            X, y, cfg.hyperparameters.labels
        )
        train_iter, dev_iter, test_iter = wrap_data(
            X_train, y_train, X_valid, y_valid, X_test, y_test, pre_embed
        )

        ### Update args and print
        if cfg.cuda and not torch.cuda.is_available():
            cfg.cuda = False
        print("\nParameters:")
        for attr, value in sorted(cfg.__dict__.items()):
            print("\t{}={}".format(attr.upper(), value))

        ### Model
        cnn = model.CNN_Text(vars(cfg.hyperparameters))
        if cfg.input.snapshot is not None:
            print("\nLoading model from {}...".format(cfg.input.snapshot))
            cnn.load_state_dict(torch.load(cfg.input.snapshot))

        ### training
        train.train(train_iter, dev_iter, test_iter, cnn, cfg)

    else:
        ### predict
        ### load sequences
        if cfg.input.data_path:
            seqs, y = load_data(cfg.input.data_path)
        else:
            print("Please provide data file for training...")
            exit(0)

        ### tokenization
        kmer_size = 1 if cfg.hyperparameters.nuc else 3
        voc_dict = build_vob(kmer_size)
        cfg.hyperparameters.vocab_size = len(voc_dict)
        X = []
        for seq in seqs:
            X.append(mytok(seq, voc_dict, kmer_size))

        ### load model
        cnn = model.CNN_Text(vars(cfg.hyperparameters))
        if cfg.input.snapshot is not None:
            print("\nLoading model from {}...".format(cfg.input.snapshot))
            cnn.load_state_dict(torch.load(cfg.input.snapshot))
        else:
            print("Please provide model file for inference...")

        ### inference
        cnn.eval()
        with torch.no_grad():
            for seq in X:
                pred = cnn(torch.tensor([seq])).squeeze().numpy()
                if len(pred) > 1:
                    print(np.argmax(pred))
                else:
                    print(pred)


if __name__ == "__main__":
    main()
