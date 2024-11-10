import argparse
import random
from itertools import cycle

import deepspeed
import numpy as np
import torch
from peft import AdaLoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils_biology import create_dataloader_from_directory

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common.utils import compute_kl, get_answer_loss


def main(args) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load models
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.normal_model_name)

    # Set up LoRA
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )

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

    train_bad_loader = create_dataloader_from_directory(args.unlearning_dataset, tokenizer, args)
    train_normal_loader = create_dataloader_from_directory(args.normal_dataset, tokenizer, args)

    model, _, _, _ = deepspeed.initialize(args=args, model=model)
    pretrained_model, _, _, _ = deepspeed.initialize(args=args, model=pretrained_model)
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
            model.save_pretrained(args.model_save_dir + "_checkpoint_" + str(idx))

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
        help="Name of the model to unlearn.",
    )
    parser.add_argument(
        "--normal_model_name",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on the normal loss.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.005, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="HuggingFaceH4/zephyr-7b-beta",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        help="Directory to save the model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save the model."
    )
    parser.add_argument('--seed', type=int, default=8888)
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=5000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--use_lora", action="store_true", help="Whether to use LoRA."
    )

    # Data preprocessing arguments.
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of unlearning."
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--drop_last", action="store_true")
    parser.add_argument('--normal_dataset', type=str, help="Path to directory of normal dataset")
    parser.add_argument('--unlearning_dataset', type=str, help="Path to directory of unlearning dataset")
    parser.add_argument("--one_per_file", action="store_true")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    assert not (args.drop_last == False and args.batch_size != 1), "Drop last must be true for batch size greater than 1"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)


# Check that it works without LoRA
# python -m LLMU.Biology.unlearn_harm --deepspeed --deepspeed_config configs/llama_ds_unlearn.json     --model_name facebook/opt-125m  --normal_model_name  facebook/opt-125m  --tokenizer  facebook/opt-125m  --model_save_dir temp  --normal_dataset  /scratch/jd5697/knowledge-erasure-Asa/Wikipedia_Biology_Unpruned_Collapsed  --unlearning_dataset  /scratch/jd5697/knowledge-erasure-Asa/Wikipedia_History_Concepts-in-Physics_Philosophy_Unpruned_Collapsed