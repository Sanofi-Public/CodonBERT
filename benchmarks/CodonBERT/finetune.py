import os
import math
import random
import pandas
import argparse
import numpy as np
from Bio import SeqIO
import json

import evaluate
from datasets import Dataset

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import *
from tokenizers.processors import BertProcessing

from transformers import TrainingArguments, Trainer, PreTrainedTokenizerFast, BertForSequenceClassification

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

######### Arguments Processing
parser = argparse.ArgumentParser(description='CodonBERT')

parser.add_argument('--task', '-t', type=str, help='downstream task name')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--pretrain', '-p', default='codonbert_models/checkpoint-1/', type=str, help='folder to pretrained model')
parser.add_argument('--random', '-r', type=int, default=42, help='random seed')

parser.add_argument('--batch', '-b', type=int, default=128, help='batch size')
parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs')
parser.add_argument('--eval_step', '-s', type=int, default=1, help='number of training steps between evaluations')

parser.add_argument('--lora',    action='store_false', help='use Lora')
parser.add_argument('--lorar',    type=int, default=32, help='Lora rank')
parser.add_argument('--lalpha',   type=int, default=32, help='Lora alpha')
parser.add_argument('--ldropout', type=float, default=0.1, help='Lora dropout')
args = parser.parse_args()


########### PEFT
if args.lora:
    from peft import LoraConfig, TaskType
    from peft import get_peft_model


######### Downstream Task Setting
num_of_labels = 1
task_name = args.task.lower()

if "rfp" in task_name:
    task_name = "mRFP Expression"
    data_path = "data/mRFP_Expression.csv"
elif "coli" in task_name:
    task_name = "E.Coli proteins"
    num_of_labels = 3
    data_path = "data/E.Coli_proteins.csv"
elif "fungal" in task_name:
    task_name = "Fungal expression"
    data_path = "data/Fungal_expression.csv"
elif "ribo" in task_name:
    task_name = "Tc-Riboswitches"
    data_path = "data/Tc-Riboswitches.csv"
elif "stab" in task_name:
    task_name = "mRNA Stability"
    data_path = "data/mRNA_Stability.csv"
elif "cov" in task_name:
    task_name = "CoV Vaccine Degradation"
    data_path = "data/CoV_Vaccine_Degradation.csv"
else:
    print("please provide an valid task name from: cov, rfp, coli, fungal, mlos, ribo")
    exit(0)

# kernal_num = args.kernal_num
# kernal_sizes = args.kernal_sizes

lr = args.lr
bs_train = args.batch
bs_test = args.batch
log_steps = args.eval_step
save_steps = args.eval_step
num_epoches = args.epochs

model_dir='%s_models' % task_name.replace(" ", "-")
os.makedirs(model_dir, exist_ok=True)

######## default Model hyperparameters (do not change)
max_length = 1024
hidden_size = 768
inter_size = 3072
num_heads = 12
num_layers = 12


######### data loading and processing
def mytok(seq, kmer_len, s):
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq)-kmer_len)+1, s):
        kmer_list.append(seq[j:j+kmer_len])
    return kmer_list


def load_data_from_csv(data_path, split):  # TODO: load compete dataset
    seqs = []
    ys = []

    skipped = 0
    df = pandas.read_csv(data_path)
    df = df.loc[df['Dataset'] == task_name]
    df = df.loc[df['Split'] == split]

    raw_seqs = df["Sequence"]
    raw_ys = df["Value"]

    total = len(raw_seqs)
    for seq, y in zip(raw_seqs, raw_ys):
        lst_tok = mytok(seq, 3, 3)
        if lst_tok:
            if len(lst_tok) > max_length - 2:
                skipped += 1
                print("Skip one sequence with length", len(lst_tok), \
                      "codons. Skipped %d seqs from total %d seqs." % (skipped, total))
                continue
            seqs.append(" ".join(lst_tok))
            if num_of_labels > 1:
                ys.append(int(float(y)))
            else:
                ys.append(float(y))
    return seqs, ys


def build_dataset(data_path):
    X_train, y_train = load_data_from_csv(data_path, "train")
    X_eval, y_eval = load_data_from_csv(data_path, "val")
    X_test, y_test = load_data_from_csv(data_path, "test")

    print("data size:", len(X_train), len(X_eval), len(X_test))

    ds_train = Dataset.from_list([{"seq": seq, "label": y} for seq, y in zip(X_train, y_train)])
    ds_eval = Dataset.from_list([{"seq": seq, "label": y} for seq, y in zip(X_eval, y_eval)])
    ds_test = Dataset.from_list([{"seq": seq, "label": y} for seq, y in zip(X_test, y_test)])

    return ds_train, ds_eval, ds_test


########### Vocabulary & Tokenizer
lst_ele = list('AUGCN')
lst_voc = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
for a1 in lst_ele:
    for a2 in lst_ele:
        for a3 in lst_ele:
            lst_voc.extend([f'{a1}{a2}{a3}'])

dic_voc = dict(zip(lst_voc, range(len(lst_voc))))
print(dic_voc)

tokenizer = Tokenizer(WordLevel(vocab=dic_voc, unk_token="[UNK]"))
tokenizer.add_special_tokens(['[PAD]','[CLS]', '[UNK]', '[SEP]','[MASK]'])
tokenizer.pre_tokenizer = Whitespace()
tokenizer.post_processor = BertProcessing(
    ("[SEP]", dic_voc['[SEP]']),
    ("[CLS]", dic_voc['[CLS]']),
)

bert_tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                               do_lower_case=False,
                                               clean_text=False,
                                               tokenize_chinese_chars=False,
                                               strip_accents=False,
                                               unk_token='[UNK]',
                                               sep_token='[SEP]',
                                               pad_token='[PAD]',
                                               cls_token='[CLS]',
                                               mask_token='[MASK]')

def encode_string(data):
    return bert_tokenizer_fast(data['seq'],
                               truncation=True,
                               padding="max_length",
                               max_length=max_length,
                               return_special_tokens_mask=True)


########### loading dataset
ds_train, ds_eval, ds_test = build_dataset(data_path)

train_loader = ds_train.map(encode_string, batched=True)
eval_loader = ds_eval.map(encode_string, batched=True)
test_loader = ds_test.map(encode_string, batched=True)


########### loading pretrained model and downstream task model
model = BertForSequenceClassification.from_pretrained(args.pretrain, num_labels=num_of_labels)
print("Loading model succesfully...")


########### lora
if args.lora:
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,
                             r=args.lorar,
                             lora_alpha=args.lalpha,
                             lora_dropout=args.ldropout,
                             use_rslora=True)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

######### Training Settings & Metrics
training_args = TrainingArguments(
    optim='adamw_torch',
    learning_rate=lr,                     # learning rate
    output_dir=model_dir,                 # output directory to where save model checkpoint
    evaluation_strategy="epoch",          # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=num_epoches,         # number of training epochs, feel free to tweak
    per_device_train_batch_size=bs_train, # the training batch size, put it as high as your GPU memory fits
    per_device_eval_batch_size=bs_test,   # evaluation batch size
    save_strategy="epoch",
    save_steps=save_steps,                # save model
    eval_steps=1,
    load_best_model_at_end=True,          # whether to load the best model (in terms of loss) at the end of training
    save_total_limit = 5,
    report_to=None
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if num_of_labels > 1:
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    else:
        logits = logits.flatten()
        labels = labels.flatten()

        try:
            pearson_corr = pearsonr(logits, labels)[0].item()
            spearman_corr = spearmanr(logits, labels)[0].item()
            return {
                "pearson": pearson_corr,
                "spearmanr": spearman_corr,
            }
        except:
            return {"pearson":0.0, "spearmanr":0.0}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader,
    eval_dataset=eval_loader,
    compute_metrics=compute_metrics
)


######### Training & Evaluation & Prediction
# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Prediction on test set
pred, _, metrics = trainer.predict(test_loader)
print(metrics)
