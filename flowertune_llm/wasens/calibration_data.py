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

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

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

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    data_files = {'train': './data/c4-train.00000-of-01024.json.gz',
                'validation': './data/c4-validation.00000-of-00008.json.gz'
    }
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    traindata = load_dataset('json',data_files=data_files['train'],split='train')
    valdata = load_dataset('json',data_files=data_files['validation'],split='train')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
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

def get_tokenizer(actual_task,calibdation_set, nsamples, seed, seqlen, tokenizer):
    traindata = calibdation_set
    task_to_keys = {
        "vicgalle/alpaca-gpt4":"text",
        "stanfordnlp/imdb":"text",
        "openai/gsm8k":"question",
        "Muennighoff/natural-instructions":"inputs"
    }
    key = task_to_keys[actual_task]
    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for i in range(nsamples):
        trainenc = tokenizer(traindata[key][i], padding='max_length',truncation=True,max_length=seqlen,return_tensors='pt')
        inp = trainenc.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100  
        trainloader.append((inp, tar))

    return trainloader, []

def get_nlu_tokenizer(task_name,calibdation_set, nsamples, seed, seqlen, tokenizer):
    traindata = calibdation_set
    task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
            "stanfordnlp/imdb":("text", None)
        }
    random.seed(seed)
    trainloader = []
    for i in range(nsamples):
        sentence1_key, sentence2_key = task_to_keys[task_name]
        if sentence2_key is None:
            trainenc = tokenizer(traindata[sentence1_key][i], padding='max_length',truncation=True,max_length=seqlen,return_tensors='pt')
        else:
            trainenc = tokenizer(traindata[sentence1_key], traindata[sentence2_key], padding='max_length',truncation=True,max_length=seqlen,return_tensors='pt')
        
        inp = trainenc.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100  
        trainloader.append((inp, tar))

    return trainloader, []

# Function to select the appropriate loader based on dataset name
def get_loaders(task_type='NLU',actual_task='', nsamples=128, seed=0, seqlen=2048, tokenizer=None,dataset=None):
    if 'wikitext2' in actual_task:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in actual_task:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if task_type == "NLU":
        return get_nlu_tokenizer(actual_task,dataset,nsamples, seed, seqlen, tokenizer)
    else:
        return get_tokenizer(actual_task,dataset,nsamples, seed, seqlen, tokenizer)