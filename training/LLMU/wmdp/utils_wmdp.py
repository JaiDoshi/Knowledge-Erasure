# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset, IterableDataset 
from transformers import DataCollatorForLanguageModeling
from functools import partial
import sys
import os 
import json
from itertools import zip_longest

def collate_dataset(batch):

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["input_ids"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def tokenization(examples, tokenizer, max_length):

    return tokenizer(examples['text'], padding=True, truncation=True, max_length=max_length)


def create_dataloader_from_dataset(tokenizer, args):

    if args.retain_dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    else:
        dataset = load_dataset(args.retain_dataset, ignore_verifications=True, split="train")

    # Using the same settings as in the original RMU code 
    dataset = dataset.filter(lambda l: len(l['text']) > 50)
    dataset = dataset.select(range(args.dataset_size))
    dataset = dataset.map(partial(tokenization, tokenizer=tokenizer, max_length=512), batched=True, remove_columns='text')
    
    dataset.set_format(
        type="torch"
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_dataset)
    
    return dataloader


def create_dataloader_wmdp(tokenizer, args):

    bio_dataset_path =  os.path.join(args.wmdp_dataset_path, 'bio-forget-corpus.jsonl')
    cyber_dataset_path = os.path.join(args.wmdp_dataset_path, 'cyber-forget-corpus.jsonl') 

    bio_file = open(bio_dataset_path, 'r').readlines()
    cyber_file = open(cyber_dataset_path, 'r').readlines()

    data_list_bio = []
    data_list_cyber = []

    for i in range(args.dataset_size // 2):
        data_list_bio.append(json.loads(bio_file[i])['text'])
        data_list_cyber.append(cyber_file[i])

    data_dict_bio = {"text": data_list_bio}
    dataset_bio = Dataset.from_dict(data_dict_bio)

    data_dict_cyber = {"text": data_list_cyber}
    dataset_cyber = Dataset.from_dict(data_dict_cyber)

    # Using the same settings as in the original RMU code
    dataset_bio = dataset_bio.map(partial(tokenization, tokenizer=tokenizer, max_length=512), batched=True, remove_columns='text')
    dataset_cyber = dataset_cyber.map(partial(tokenization, tokenizer=tokenizer, max_length=768), batched=True, remove_columns='text')

    dataset = Dataset.from_list([item for pair in zip_longest(dataset_bio, dataset_cyber) for item in pair if item is not None])

    dataset.set_format(
        type="torch"
    )

    # Batch size is fixed to 1 because the bio and cyber subsets are of different lengths
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_dataset)
    
    return dataloader
