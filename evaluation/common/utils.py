import json
import random
import os
import time
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import AutoPeftModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load(ckpt_dir, peft_model, tokenizer_path):

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path
    )

    tokenizer.pad_token_id = (
            0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        )

    if not peft_model:

        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, torch_dtype=torch.float16
        )

    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float16,
            attn_implementation="eager"
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()

    model.to(device)
    model.eval()

    return model, tokenizer


def prepare_input(tokenizer, prompts):
    
    input_tokens = tokenizer.encode_plus(
        prompts, return_tensors="pt", padding=True
    )
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(device)

    return input_tokens

def make_inference(model, tokenizer, prompts):
    
    answers = []

    for prompt in prompts:
        encode_inputs = prepare_input(tokenizer, prompt)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(
            outputs, skip_special_tokens=True))

    answers = [answer[-1] for answer in answers]
    return answers