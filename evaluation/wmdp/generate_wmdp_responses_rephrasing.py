import argparse
import json
import os
from os import path, sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import pandas as pd
from common.utils import data_directory_list, gen_prompt, load, make_inference
from wmdp_utils import TASKS, format_wmdp_example


def main(args):

    run_results = {}
    for directory in data_directory_list:
        run_results[directory] = {}

    if args.dev_task == "":
        output_breakpoint_name = "run_breakpoint_%s_rephrasing.json" % (
            args.extra_info)
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

    def load_df():

        return pd.read_csv(
            os.path.join(args.MMLU_dir, 'dev', args.dev_task + "_dev.csv"), header=None
        )[: args.ntrain]

    def load_json(task):
        f = open(os.path.join(args.data_dir, directory, task + '.json'))
        return json.load(f)

    print(
        f"Processing {directory}"
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

        test_data = load_json(task)
        dev_df = load_df() if args.dev_task != "" else None

        for question in test_data:

            k = args.ntrain
            prompt_end = format_wmdp_example(question)

            train_prompt = gen_prompt(
                dev_df, k
            ) if args.dev_task != "" else ""

            prompt = args.system_prompt + train_prompt + prompt_end

            label = chr(65 + question['answer'])

            prompts.append(prompt)
            labels.append(label)

        pred_answers = make_inference(
            model, tokenizer, prompts
        )

        run_results[directory][task] = {
            "pred_answers": pred_answers, "gold_answers": labels}
        json.dump(run_results, open(output_breakpoint_name, "w"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, help="Path to the model")
    parser.add_argument("--peft_model", action="store_true",
                        help="Whether the model is a PEFT model")
    parser.add_argument("--data_dir", type=str, help="Path to wmdp dataset")
    parser.add_argument("--MMLU_dir", type=str,
                        help="Path to MMLU dataset for n-shot prompting")
    parser.add_argument("--extra_info", type=str, default="")
    parser.add_argument("--ntrain", type=int, default=0,
                        help="Number of examples to use for n-shot prompting")
    parser.add_argument("--tokenizer", type=str,
                        default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--dev_task", type=str, default="",
                        help="MMLU task to be used for n-shot prompting (e.g. college_chemistry)")
    parser.add_argument("--system_prompt", type=str,
                        default="The following are multiple choice questions (with answers).\n\n")
    args = parser.parse_args()

    config = vars(args)
    print("====Config====")
    print("\n".join(f"{k}={v}" for k, v in config.items()))
    print("=====")

    main(args)
