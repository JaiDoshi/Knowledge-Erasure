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
import sys

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common.utils import load, make_inference


# data_directory_list = ['data_filler_1', 'data_filler_2', 'data_filler_before_1', 'data_filler_before_2', 
#                         'data_filler_before_3',
#                         'data_rephrased_conversation_2', 
#                        'data_rephrased_poem_1', 'data_rephrased_remove_technical_1', 'data_rephrased_remove_technical_2', 
#                        'data_translated_french_2', 'data_translated_german_2', 'data_translated_hindi_2',
#                        'data_translated_korean_2', 'data_translated_arabic_2', 'data_translated_czech_2',
#                         'data_translated_bengali_2', 'data_translated_vietnamese_2', 'data_translated_turkish_2',
#                        'data_translated_telugu_2', 'data_translated_farsi_2',
#                        'data_with_variables_2', 'data_filler_before_latin_rephrased_conversation', 'data_filler_before_english_rephrased_conversation',
#                        'data_filler_before_hindi_rephrased_conversation', 'data_filler_before_latin_rephrased_poem', 'data_filler_before_english_rephrased_poem',
#                        'data_filler_before_hindi_rephrased_poem'
#                        ]


data_directory_list = ['data_filler_1', 'data_rephrased_remove_technical_1', 'data_translated_farsi_2'
                       ]

def main(args):

    run_results = {}
    for directory in data_directory_list:
        run_results[directory] = {}

    if args.dev_task == "":
        output_breakpoint_name = "run_breakpoint_%s_rephrasing.json" % (args.extra_info)
        output_filename = "run_results_%s_rephrasing.json" % (args.extra_info)
    else:
        output_breakpoint_name = "run_breakpoint_%s_%sdev_rephrasing.json" % (
            args.extra_info, args.dev_task)
        output_filename = "run_results_%s_%sdev_rephrasing.json" % (
            args.extra_info, args.dev_task)
    if os.path.isfile(output_breakpoint_name):
        run_results = json.load(open(output_breakpoint_name))

    model, tokenizer = load(args.ckpt_dir, args.peft_model, args.tokenizer)

    for directory in data_directory_list:
        generate_results_for_prompt(
            args,
            model,
            tokenizer,
            run_results,
            output_breakpoint_name,
            directory
        )

    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)


def generate_results_for_prompt(
    args, model, tokenizer, run_results, output_breakpoint_name, directory
):

    def load_df(task, split="dev"):

        if split == "test":
            return pd.read_csv(
                os.path.join(args.data_dir, directory, split, task + "_test.csv"), header=None
            )
        else:
            if args.dev_task == "":
                return pd.read_csv(
                    os.path.join(args.data_dir, directory, split, task + "_dev.csv"), header=None
                )[: args.ntrain]
            else:
                return pd.read_csv(
                    os.path.join(args.data_dir, directory, split, args.dev_task + "_dev.csv"), header=None
                )[: args.ntrain]

    print(
        f"Generating responses with {args.ntrain} few-shot examples for each task in {directory}"
    )
    for task in TASKS:
        if (
            task in run_results[directory]
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

        run_results[directory][task] = {"pred_answers": pred_answers, "gold_answers": labels}
        json.dump(run_results, open(output_breakpoint_name, "w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--peft_model", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True)
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
