#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   calibration_data.py
@Time    :   2024/12/28 16:28:20
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   data utils of calibration for mausuring. also fork from wanda.
'''

# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

from pathlib import Path

import numpy as np
import random
import torch
from datasets import load_dataset

from project_config import load_project_config, require_local_path

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def _load_local_text(path_value, field):
    path = Path(require_local_path(path_value, field))
    if not path.is_file():
        raise FileNotFoundError(f"{field} must be a local file: {path}")
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if suffixes[-1:] in [[".json"], [".jsonl"]] or suffixes[-2:] in [
        [".json", ".gz"],
        [".jsonl", ".gz"],
    ]:
        dataset = load_dataset("json", data_files=str(path), split="train")
    elif suffixes[-1:] in [[".txt"], [".text"]] or suffixes[-2:] in [
        [".txt", ".gz"],
        [".text", ".gz"],
    ]:
        dataset = load_dataset("text", data_files=str(path), split="train")
    else:
        raise ValueError(
            f"Unsupported local calibration format for {field}: {path.name}"
        )
    if "text" not in dataset.column_names:
        raise ValueError(f"{field} must provide a 'text' column.")
    return dataset


# Load and process local Wikitext2 data.
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    cfg = load_project_config().llm.calibration.wikitext2
    traindata = _load_local_text(
        cfg.train_path, "llm.calibration.wikitext2.train_path"
    )
    testdata = _load_local_text(
        cfg.test_path, "llm.calibration.wikitext2.test_path"
    )

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if trainenc.input_ids.shape[1] <= seqlen:
        raise ValueError(
            "The local Wikitext2 training file does not contain enough tokens "
            f"for seqlen={seqlen}."
        )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process local C4 data.
def get_c4(nsamples, seed, seqlen, tokenizer):
    cfg = load_project_config().llm.calibration.c4
    traindata = _load_local_text(cfg.train_path, "llm.calibration.c4.train_path")
    valdata = _load_local_text(
        cfg.validation_path, "llm.calibration.c4.validation_path"
    )

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        for _attempt in range(1000):
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        else:
            raise ValueError(
                "The local C4 training file has no sampled records longer than "
                f"seqlen={seqlen}."
            )
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def find_first_common(list1, list2):
    set2 = set(list2)
    for item in list1:
        if item in set2:
            return item
    return None  

def get_tokenizer(
    calibdation_set, nsamples, seed, seqlen, tokenizer, text_column
):
    traindata = calibdation_set
    if not text_column:
        raise ValueError("A local calibration text column is required.")
    if text_column not in traindata.column_names:
        raise ValueError(
            f"Calibration dataset is missing the '{text_column}' column."
        )
    if len(traindata) < nsamples:
        raise ValueError(
            f"Calibration requires {nsamples} rows, but only {len(traindata)} are available."
        )
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for i in range(nsamples):
        trainenc = tokenizer(traindata[text_column][i], padding='max_length',truncation=True,max_length=seqlen,return_tensors='pt')
        inp = trainenc.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100  
        trainloader.append((inp, tar))

    return trainloader, []

# Function to select the appropriate loader based on dataset name
def get_loaders(task_type='NLU',actual_task='', nsamples=128, seed=0, seqlen=2048, tokenizer=None,dataset=None, text_column=None):
    if 'wikitext2' in actual_task:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in actual_task:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if task_type == "NLU":
        raise ValueError("Only NLG calibration with local datasets is supported.")
    return get_tokenizer(dataset, nsamples, seed, seqlen, tokenizer, text_column)
