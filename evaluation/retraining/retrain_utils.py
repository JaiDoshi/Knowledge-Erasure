from functools import partial

import torch
from datasets import load_dataset


def collate_dataset(batch):

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# We pad to multiples of max_length so that in the next step we can split the input into chunks of size max_length
def tokenization(examples, tokenizer, args):

    tokens = tokenizer(examples['text'])['input_ids']

    def get_padding_length(length, multiple):
        return multiple - length % multiple

    padding_length = get_padding_length(len(tokens), args.max_length)

    return tokenizer(examples['text'], padding='max_length', max_length=len(tokens) + padding_length)


def create_dataset(encoding, args):

    stride = args.stride
    max_length = args.max_length

    results = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(encoding['input_ids'])):
        seq_len = len(encoding['input_ids'][i])

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):

            assert (begin_loc + max_length <= seq_len)

            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encoding['input_ids'][i][begin_loc:end_loc]
            attention_mask = encoding['attention_mask'][i][begin_loc:end_loc]
            target_ids = input_ids.copy()

            target_ids[:-trg_len] = [-100]*(len(target_ids)-trg_len)
            
            # Set the target_ids to -100 where the attention mask is 0 (corresponding to padding tokens)
            target_ids = [-100 if mask == 0 else target for target,
                          mask in zip(target_ids, attention_mask)]

            results['input_ids'].append(torch.tensor(input_ids))
            results['labels'].append(torch.tensor(target_ids))
            results["attention_mask"].append(torch.tensor(attention_mask))

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

            if args.one_per_example:
                break

    return results


def create_dataset_from_huggingface(dataset, tokenizer, args):

    dataset = load_dataset(
        dataset, verification_mode='no_checks', split="train")

    dataset = dataset.map(partial(tokenization, tokenizer=tokenizer,
                          args=args), batched=False, remove_columns='text')
    dataset = dataset.map(partial(create_dataset, args=args), batched=True)
    dataset.set_format("pt")

    return dataset
