from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing


def mytok(seq, kmer_len, s):
    """
    Tokenize a sequence into kmers.

    Args:
        seq (str): sequence to tokenize
        kmer_len (int): length of kmers
        s (int): increment
    """
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq) - kmer_len) + 1, s):
        kmer_list.append(seq[j : j + kmer_len])
    return kmer_list


def get_tokenizer():
    """Create tokenizer."""
    lst_ele = list("AUGCN")
    lst_voc = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    for a1 in lst_ele:
        for a2 in lst_ele:
            for a3 in lst_ele:
                lst_voc.extend([f"{a1}{a2}{a3}"])
    dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
    tokenizer = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
    tokenizer.add_special_tokens(["[PAD]", "[CLS]", "[UNK]", "[SEP]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = BertProcessing(
        ("[SEP]", dic_voc["[SEP]"]),
        ("[CLS]", dic_voc["[CLS]"]),
    )
    return tokenizer
