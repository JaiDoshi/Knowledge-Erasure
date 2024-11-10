import json
import random
import os
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
from generation_utils import format_example, gen_prompt, TASKS

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common.utils import load, make_inference

def main(args):

    run_results = {}

    if args.dev_task == "":
        output_breakpoint_name = "run_breakpoint_%s.json" % (args.extra_info)
        output_filename = "run_results_%s.json" % (args.extra_info)
    else:
        output_breakpoint_name = "run_breakpoint_%s_%sdev.json" % (
            args.extra_info, args.dev_task)
        output_filename = "run_results_%s_%sdev.json" % (
            args.extra_info, args.dev_task)
    if os.path.isfile(output_breakpoint_name):
        run_results = json.load(open(output_breakpoint_name))

    model, tokenizer = load(args.ckpt_dir, args.peft_model, args.tokenizer)

    generate_results_for_prompt(
        args,
        model,
        tokenizer,
        run_results,
        output_breakpoint_name
    )

    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)


def generate_results_for_prompt(
    args, model, tokenizer, run_results, output_breakpoint_name
):

    def load_df(task, split="dev"):

        if split == "test":
            return pd.read_csv(
                os.path.join(args.data_dir, split, task + "_test.csv"), header=None
            )
        else:
            if args.dev_task == "":
                return pd.read_csv(
                    os.path.join(args.data_dir, split, task + "_dev.csv"), header=None
                )[: args.ntrain]
            else:
                return pd.read_csv(
                    os.path.join(args.data_dir, split, args.dev_task + "_dev.csv"), header=None
                )[: args.ntrain]

    print(
        f"Generating responses with {args.ntrain} few-shot examples for each task"
    )

    for task in TASKS:
        if (
            task in run_results
        ):
            print("Skipping %s ..." % task)
            continue
        print("Testing %s ..." % task)

        prompts = []
        labels = []

        dev_df = load_df(task)
        test_df = load_df(task, split="test")

        for i in range(test_df.shape[0]):

            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)

            train_prompt = gen_prompt(
                dev_df, k
            )

            prompt = args.system_prompt + train_prompt + prompt_end
            label = test_df.iloc[i, test_df.shape[1] - 1]

            prompts.append(prompt)
            labels.append(label)
             
        pred_answers = make_inference(
            model, tokenizer, prompts
        )

        run_results[task] = {
            "pred_answers": pred_answers, "gold_answers": labels}
        json.dump(run_results, open(output_breakpoint_name, "w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--peft_model", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--extra_info", type=str, default="")
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--tokenizer", type=str,
                        default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--dev_task", type=str, default="", help="Task to be used for n-shot prompting (e.g. college_chemistry). If not provided, the dev set of subject being evaluated will be used.")
    parser.add_argument("--system_prompt", type=str,
                        default="The following are multiple choice questions (with answers).\n\n")
    args = parser.parse_args()

    config = vars(args)
    print("====Config====")
    print("\n".join(f"{k}={v}" for k, v in config.items()))
    print("=====")

    main(args)
