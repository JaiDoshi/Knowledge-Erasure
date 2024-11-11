import argparse
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    return results


def evaluate(model, dataloader):

    total_nll = torch.tensor([0]).float()
    total_num_datapoints = torch.tensor([0])
    total_num_correctly_predicted = torch.tensor([0])

    for item in dataloader:

        input_ids = item['input_ids'].to(device)
        attention_mask = item['attention_mask'].to(device)
        labels = item['labels'].to(device)

        with torch.no_grad():

            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)

            shifted_logits = outputs.logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()

            labels_to_predict = shifted_labels != -100
            num_datapoints = torch.sum(labels_to_predict)
            predictions = torch.argmax(shifted_logits, dim=-1)
            num_correctly_predicted = torch.sum(
                (predictions == shifted_labels) * labels_to_predict)

            loss_fn = torch.nn.CrossEntropyLoss(
                ignore_index=-100, reduction='sum')
            loss = loss_fn(
                shifted_logits.view(-1, shifted_logits.shape[-1]), shifted_labels.view(-1))

            total_nll += loss.to('cpu')
            total_num_datapoints += num_datapoints.to('cpu')
            total_num_correctly_predicted += num_correctly_predicted.to('cpu')

    return total_nll, total_num_datapoints, total_num_correctly_predicted


def evaluate_from_dataset(args, model, tokenizer):

    dataset = load_dataset(
        args.dataset, ignore_verifications=True, split="train")

    dataset = dataset.train_test_split(
        train_size=args.proportion, shuffle=True)['train']
    dataset = dataset.map(
        partial(tokenization, tokenizer=tokenizer), batched=True, remove_columns='text')
    dataset = dataset.map(partial(create_dataset, args=args), batched=True)
    dataset.set_format(type="torch")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_dataset)
    total_nll, total_num_datapoints, total_num_correctly_predicted = evaluate(
        model, dataloader)

    print("Number of data points:", total_num_datapoints)
    print("Number correctly predicted:", total_num_correctly_predicted)
    print("Accuracy:", total_num_correctly_predicted/total_num_datapoints)
    print("Perplexity:", torch.exp(total_nll/total_num_datapoints))


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager")
    model.to(device)

    evaluate_from_dataset(args, model, tokenizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="stas/openwebtext-10k",
                        help="HuggingFace dataset for perplexity evaluation")
    parser.add_argument("--proportion", type=float, default=1,
                        help="Proportion of the dataset to use for evaluation")
    parser.add_argument("--tokenizer", type=str,
                        default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--model", type=str,
                        default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--max_length", type=int,
                        default=256, help="Size of each chunk")
    parser.add_argument("--stride", type=int, default=128,
                        help="Stride size to use during chunking")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size of evaluation")
    parser.add_argument("--drop_last", action="store_true",
                        help="Whether to drop the last chunk in the text if it is less than max_length. Must be true for batch size greater than 1")
    args = parser.parse_args()

    assert not (args.drop_last is False and args.batch_size !=
                1), "Drop last must be true for batch size greater than 1"

    main(args)
