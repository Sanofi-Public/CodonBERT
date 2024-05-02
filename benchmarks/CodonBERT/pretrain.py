import os
import random
import evaluate
import numpy as np
from Bio import SeqIO
from datasets import Dataset, IterableDataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import *
from tokenizers.processors import BertProcessing

from transformers import TrainingArguments, Trainer, PreTrainedTokenizerFast, BertConfig, DataCollatorForLanguageModeling, BertForPreTraining

import torch
import pandas as pd


########## input/output path
train_data_path = 'data/train_samples.csv'
val_data_path = 'data/eval.csv'

model_dir='codonbert_models'
os.makedirs(model_dir, exist_ok=True)


########## hyper-parameters
max_length = 1024  # default
hidden_size = 768  # default
inter_size = 3072  # default
num_heads = 12     # default
num_layers = 12    # default

bs_train = 32
bs_test =  32

num_epoches = 1
log_steps = 10
save_steps = 10

########## funs for loading sequences
def mytok(seq, kmer_len, s):
    seq = seq.upper().replace("T", "U")
    kmer_list = []
    for j in range(0, (len(seq)-kmer_len)+1, s):
        kmer_list.append(seq[j:j+kmer_len])
    return kmer_list


def seq2dataset_iter():
    for i_epoch in range(num_epoches):
        df = pd.read_csv(train_data_path)
        df = df.sample(frac=1)
        for index, row in df.iterrows():
            seq1 = " ".join(mytok(row["Seq1"], 3, 3)[:510])
            seq2 = " ".join(mytok(row["Seq2"], 3, 3)[:510])
            yield {"seq0": seq1, "seq1": seq2, "next_sentence_label": int(row["STP"])}



def seq2dataset_iter2():
    df = pd.read_csv(val_data_path).iloc[:100]
    for index, row in df.iterrows():
        seq1 = " ".join(mytok(row["Seq1"], 3, 3)[:510])
        seq2 = " ".join(mytok(row["Seq2"], 3, 3)[:510])
        yield {"seq0": seq1, "seq1": seq2, "next_sentence_label": int(row["STP"])}


########## building vocs for tokenizers
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

########## tokenizer
pre_tok_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                       do_lower_case=False,
                                       clean_text=False,
                                       tokenize_chinese_chars=False,
                                       strip_accents=False,
                                       unk_token='[UNK]',
                                       sep_token='[SEP]',
                                       pad_token='[PAD]',
                                       cls_token='[CLS]',
                                       mask_token='[MASK]')

bert_tokenizer_fast = pre_tok_fast
bert_tokenizer = pre_tok_fast

########## loading and tokenizer sequences
def encode_string(data):
    return bert_tokenizer_fast(data['seq0'],
                               data['seq1'],
                               truncation=True,
                               padding="max_length",
                               max_length=max_length,
                               return_special_tokens_mask=True)


##### loading seqs
print('loading train seqs')
train_generator = IterableDataset.from_generator(seq2dataset_iter)
print('loading val seqs')
eval_generator = IterableDataset.from_generator(seq2dataset_iter2)


###### tokenizer and padding seeqs
print('padding train')
train_padded_generator = train_generator.map(encode_string, batched=True)
print('padding val')
eval_padded_generator = eval_generator.map(encode_string, batched=True)


######### Model Config
vocs = tokenizer.get_vocab()
model_config = BertConfig(vocab_size=len(vocs),
                          max_position_embeddings=max_length,
                          num_hidden_layers=num_layers,
                          num_attention_heads=num_heads,
                          hidden_size=hidden_size,
                          intermediate_size=inter_size)
model = BertForPreTraining(config=model_config)


########## evaluation
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    mlm_logits, cls_logits = logits
    mlm_labels, cls_labels = labels

    # mlm acc
    mlm_preds = np.argmax(mlm_logits, axis=-1)
    labels1 = mlm_labels.reshape((-1,))
    pred1 = mlm_preds.reshape((-1,))
    idx = labels1>=0
    labels2 = labels1[idx]
    pred2 = pred1[idx]

    mlm_acc = metric.compute(predictions=pred2, references=labels2)['accuracy']

    # seq relationship cls
    cls_preds = np.argmax(cls_logits, axis=-1)
    labels1 = cls_labels.reshape((-1,))
    pred1 = cls_preds.reshape((-1,))

    seq_relationship_acc = metric.compute(predictions=pred1, references=labels1)['accuracy']

    return {"mlm_acc": mlm_acc,
            "seq_relationship_acc": seq_relationship_acc}

########## Training Config
training_args = TrainingArguments(
    optim='adamw_torch',
    learning_rate=0.0001,                 # learning rate
    warmup_steps=10000,
    weight_decay=0.01,
    max_steps=10,                         # added specifally for IterableDataset
    num_train_epochs=num_epoches,         # number of training epochs, feel free to tweak
    per_device_train_batch_size=bs_train, # the training batch size, put it as high as your GPU memory fits
    per_device_eval_batch_size=bs_test,   # evaluation batch size
    evaluation_strategy="steps",          # evaluate each `logging_steps` steps
    eval_steps=log_steps,                 # evaluate model
    logging_steps=log_steps,              # log model
    save_steps=save_steps,                # save model
    load_best_model_at_end=True,          # whether to load the best model (in terms of loss) at the end of training
    output_dir=model_dir,                 # output directory to where save model checkpoint
    overwrite_output_dir=True,
    save_total_limit = 10
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=bert_tokenizer, mlm=True, mlm_probability=0.15
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_padded_generator,
    eval_dataset=eval_padded_generator,
    compute_metrics=compute_metrics
)

########### Train
trainer.train()
