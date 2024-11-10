import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import os

def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_data(forget_corpora, retain_corpora, wmdp_dataset_path, min_len=50, max_len=2000, batch_size=4):

    def get_dataset(name):
        data = []
        if name == "wikitext":
            from datasets import load_dataset
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            for line in open(os.path.join(wmdp_dataset_path, name + ".jsonl"), "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )