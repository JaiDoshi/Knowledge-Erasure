import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

from datasets import load_dataset, IterableDataset 
from functools import partial
import os 
import sys
import pickle

from peft import AutoPeftModelForCausalLM

def load_model(model_name_or_path, tokenizer_path, peft_model):

    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if peft_model:
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        model = peft_model.merge_and_unload()
        for param in model.parameters():
            param.requires_grad = True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )       

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def collate_dataset(batch):

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def tokenization(examples, tokenizer):

    return tokenizer(examples['text'])


def create_dataset(encoding, args):

    stride = args.stride
    max_length = args.max_length
    drop_last = args.drop_last

    results = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(encoding['input_ids'])):
        seq_len = len(encoding['input_ids'][i])
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):

            if begin_loc + max_length > seq_len and drop_last:
                break 

            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encoding['input_ids'][i][begin_loc:end_loc]
            attention_mask = encoding['attention_mask'][i][begin_loc:end_loc]
            target_ids = input_ids.copy()
            target_ids[:-trg_len] = [-100]*(len(target_ids)-trg_len)

            results['input_ids'].append(torch.tensor(input_ids))
            results['labels'].append(torch.tensor(target_ids))
            results["attention_mask"].append(torch.tensor(attention_mask))

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

            if args.one_per_file:
                break


    return results


def create_dataloader_from_directory(directory, tokenizer, args):

    def file_generator(directory):

        files = os.listdir(directory)
        random.shuffle(files)

        for file in files:
            if not file.endswith('.txt'):
                continue
            file = os.path.join(directory, file)
     
            with open(file, "r") as f:
                yield {'text': f.read()}
    
    dataset = IterableDataset.from_generator(file_generator, gen_kwargs={"directory": directory})

    dataset = dataset.map(partial(tokenization, tokenizer=tokenizer), batched=True, remove_columns='text')
    dataset = dataset.map(partial(create_dataset, args=args), batched=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_dataset)

    return dataloader


def get_data(tokenizer, args):
 
    return (
        [create_dataloader_from_directory(args.forget_corpus, tokenizer, args)],
        [create_dataloader_from_directory(args.retain_corpus, tokenizer, args)]
    )