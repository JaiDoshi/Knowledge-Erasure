import json
import os
import random
from itertools import cycle
from os import path, sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
import torch
import tqdm as tqdm
from common.utils import forward_with_cache, get_params
from transformers import AdamW
from utils_biology import get_data, load_model


def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args
):
    rmu_config = vars(args)
    print("====rmu Config====")
    print("\n".join(f"{k}={v}" for k, v in rmu_config.items()))
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    frozen_module = eval(
        args.module_str.format(model_name="frozen_model",
                               layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(
            model_name="updated_model", layer_id=args.layer_id)
    )

    control_vectors_list = []

    for i in range(len(forget_data_list)):
        random_vector = torch.rand(1, 1, updated_model.config.hidden_size,
                                   dtype=updated_model.dtype, device=updated_model.device)
        control_vec = random_vector / \
            torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    path = args.output_dir
    num_steps = 0

    for epoch in range(args.num_epochs):
        print(f"======= Epoch {epoch} =======")

        for idx, (unlearn_batch, retain_batch) in enumerate(zip(cycle(forget_data_list[0]), cycle(retain_data_list[0]))):

            topic_idx = idx % len(forget_data_list)

            unlearn_batch = {k: v.to(updated_model.device)
                             for k, v in unlearn_batch.items()}
            retain_batch = {k: v.to(updated_model.device)
                            for k, v in retain_batch.items()}

            updated_forget_activations = forward_with_cache(
                updated_model, unlearn_batch, module=updated_module, no_grad=False
            ).to(updated_model.device)

            control_vec = control_vectors_list[topic_idx]

            unlearn_loss = torch.nn.functional.mse_loss(
                updated_forget_activations, control_vec
            )

            # Retain loss
            updated_retain_activations = forward_with_cache(
                updated_model, retain_batch, module=updated_module, no_grad=False
            ).to(updated_model.device)
            frozen_retain_activations = forward_with_cache(
                frozen_model, retain_batch, module=frozen_module, no_grad=True
            ).to(updated_model.device)

            retain_loss = torch.nn.functional.mse_loss(
                updated_retain_activations, frozen_retain_activations
            )
            retain_loss *= args.alpha[topic_idx]

            # Update model
            loss = unlearn_loss + retain_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"loss: {loss.item():.4g} | unlearn_loss: {unlearn_loss.item():.4g} | retain_loss: {retain_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")

            # ======= Logging ======
            if args.verbose:

                frozen_forget_activations = forward_with_cache(
                    frozen_model, unlearn_batch, module=frozen_module, no_grad=True).to(updated_model.device)
                unlearn_cosine = torch.nn.functional.cosine_similarity(
                    updated_forget_activations, frozen_forget_activations, dim=-1).mean()
                retain_cosine = torch.nn.functional.cosine_similarity(
                    updated_retain_activations, frozen_retain_activations, dim=-1).mean()

                print(f"Step number={num_steps}")
                print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
                print(f"retain_cosine_sim={retain_cosine.item()}")
                print(f"Topic {topic_idx} updated_forget_activations.norm=", torch.mean(
                    updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                print(f"Topic {topic_idx} frozen_forget_activations.norm=", torch.mean(
                    frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                print(f"Topic {topic_idx} updated_retain_activations.norm=", torch.mean(
                    updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())
                print(f"Topic {topic_idx} frozen_retain_activations.norm=", torch.mean(
                    frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item())

            num_steps += 1

            if num_steps % args.save_every == 0:
                updated_model.save_pretrained(
                    path + "_checkpoint_" + str(num_steps))

            if num_steps == args.max_unlearn_steps:
                break

    # Save model
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(f"Saved model to {path}")


def get_args():

    import argparse
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument(
        "--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="Model to peform unlearning on"
    )
    parser.add_argument("--peft_model", action="store_true",
                        help="Is the model to peform unlearning on a PEFT model")
    parser.add_argument(
        "--normal_model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="Normal model. It is not a PEFT model"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="HuggingFaceH4/zephyr-7b-beta"
    )
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.model.layers[{layer_id}]"
    )

    # Data arguments
    parser.add_argument(
        "--retain_corpus", type=str, help="Path to the directory of the retain corpus"
    )
    parser.add_argument(
        "--forget_corpus", type=str, help="Path to the directory of the forget corpus"
    )
    parser.add_argument("--max_length", type=int,
                        default=1024, help="Size of each chunk")
    parser.add_argument("--stride", type=int, default=1024,
                        help="Stride size to use during chunking")
    parser.add_argument("--drop_last", action="store_true",
                        help="Whether to drop the last chunk in the file if it is less than max_length. Must be true for batch size greater than 1")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size of unlearning")
    parser.add_argument("--one_per_file", action="store_true",
                        help="Whether to create only one chunk per file")

    # rmu hyperparameters
    parser.add_argument("--alpha", type=str, default="100",
                        help="Retain weight")
    parser.add_argument("--steering_coeff", type=str,
                        default="20", help="Steer vector weight")
    parser.add_argument("--layer_id", type=int, default=7,
                        help="Layer to unlearn")
    parser.add_argument("--layer_ids", type=str,
                        default="5,6,7", help="Update layers")
    parser.add_argument("--param_ids", type=str,
                        default="6", help="Update params")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=50,
                        help="How many steps to checkpoint the model")
    parser.add_argument("--max_unlearn_steps", type=int,
                        default=1000, help="Max number of unlearning steps")
    parser.add_argument("--seed", type=int, default=8888, help="Seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Logging the activations norms and cosine at each step")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Dieectory to save the model")

    args = parser.parse_args()

    assert not (args.drop_last is False and args.batch_size !=
                1), "Drop last must be true for batch size greater than 1"

    args.steering_coeff_list = [float(args.steering_coeff)]
    args.alpha = [float(args.alpha)]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    return args


if __name__ == "__main__":

    args = get_args()

    SEED = args.seed
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    frozen_model, tokenizer = load_model(
        args.normal_model_name_or_path, args.tokenizer_path, False)
    updated_model, tokenizer = load_model(
        args.model_name_or_path, args.tokenizer_path, args.peft_model)

    forget_data_list, retain_data_list = get_data(
        tokenizer, args
    )

    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )
