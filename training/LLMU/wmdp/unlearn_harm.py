import argparse
import random
import sys
from itertools import cycle
from os import path, sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import deepspeed
import numpy as np
import torch
from common.utils import compute_kl, get_answer_loss
from peft import AdaLoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_wmdp import create_dataloader_from_dataset, create_dataloader_wmdp


def main(args) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        args.normal_model_name)

    # Set up LoRA
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
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

        model = get_peft_model(model, peft_config)
        model.config.use_cache = False

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    train_bad_loader = create_dataloader_wmdp(tokenizer, args)
    train_normal_loader = create_dataloader_from_dataset(tokenizer, args)

    model, _, _, _ = deepspeed.initialize(args=args, model=model)
    pretrained_model, _, _, _ = deepspeed.initialize(
        args=args, model=pretrained_model)
    model.train()

    # Start unlearning.
    idx = 0

    for bad_batch, normal_batch in zip(cycle(train_bad_loader), cycle(train_normal_loader)):

        ############ GA on answer only. ############
        bad_loss = get_answer_loss("ga", bad_batch, model, device=device)

        ############ KL on normal samples. ############
        normal_loss = compute_kl(pretrained_model, model, normal_batch, device)

        # Final loss = bad loss + normal loss
        loss = (
            args.bad_weight * bad_loss
            + args.normal_weight * normal_loss
        )

        model.backward(loss)
        model.step()

        # Print
        stats = (
            f"batch: {idx}, "
            f"bad_loss: {-bad_loss}, "
            f"current_div_loss: {normal_loss}, "
        )
        print(stats)

        idx += 1
        if idx % args.save_every == 0:
            model.save_pretrained(args.model_save_dir +
                                  "_checkpoint_" + str(idx))

        if idx == args.max_unlearn_steps:
            break

    # Save final model
    model.save_pretrained(args.model_save_dir)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Name of the model to unlearn",
    )
    parser.add_argument(
        "--normal_model_name",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Name of the pretrained model",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on the normal loss",
    )
    parser.add_argument(
        "--bad_weight",
        type=float,
        default=0.005,
        help="Weight on the bad loss"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        help="Directory to save the model",
    )
    parser.add_argument(
        "--save_every",
        type=int, default=500, help="How many steps to save model."
    )
    parser.add_argument(
        "--retain_dataset",
        type=str,
        default="wikitext",
        help="Retain dataset for to use. Must be a Huggingface dataset"
    )
    parser.add_argument(
        "--wmdp_dataset_path",
        type=str
    )
    parser.add_argument('--seed',
                        type=int,
                        default=8888)
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=5000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether to use LoRA."
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=2000
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
