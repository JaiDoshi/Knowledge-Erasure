import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from retrain_utils import collate_dataset, create_dataset_from_huggingface
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    print('Generating dataset...')
    dataset = create_dataset_from_huggingface(args.dataset, tokenizer, args)

    print('Setting up model...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16)

    if args.use_lora:
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "o_proj", "k_proj",
                            "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Convert model to bfloat16
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.bfloat16)
            if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
                if hasattr(module, "weight"):
                    module = module.to(torch.bfloat16)

        model = get_peft_model(model, lora_config)

    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.model_save_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        save_steps=args.save_every,
        seed=args.seed,
        logging_steps=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_dataset,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    print('Training model...')
    trainer.train()

    df = pd.DataFrame(trainer.state.log_history)
    df.to_csv(f'{args.model_save_dir}/loss_history.csv')

    print('Saving model...')
    model.save_pretrained(f'{args.model_save_dir}')


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='Model to retrain')
    parser.add_argument('--tokenizer', type=str,
                        default='HuggingFaceH4/zephyr-7b-beta')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--use-lora", action='store_true')
    parser.add_argument("--save_every", type=int, default=100, help="How many steps to save the model"
                        )
    parser.add_argument("--model_save_dir", type=str,
                        help="Directory to save the model")
    parser.add_argument('--seed', type=int, default=8888)
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Max number of retraining steps")

    # Data arguments
    parser.add_argument('--dataset', type=str, default='stas/openwebtext-10k')
    parser.add_argument("--max_length", type=int,
                        default=1024, help="Size of each chunk")
    parser.add_argument("--stride", type=int, default=512,
                        help="Stride size to use during chunking")
    parser.add_argument("--one_per_example", action="store_true",
                        help="Whether to create only one chunk per example")

    args = parser.parse_args()

    assert (args.max_length % args.stride ==
            0), "max_length should be a multiple of stride"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.model_save_dir, exist_ok=True)

    print(vars(args))

    with open(f'{args.model_save_dir}/argsCLI.json', 'w') as f:
        json.dump(vars(args), f)

    main(args)
