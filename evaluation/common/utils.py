import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

choices = ["A", "B", "C", "D"]

data_directory_list = ['data_latin_filler_text', 'data_english_filler_text', 'data_hindi_filler_text', 'data_rephrased_conversation',
                       'data_rephrased_poem', 'data_technical_terms_removed_1', 'data_technical_terms_removed_2', 'data_translated_french',
                       'data_translated_german', 'data_translated_hindi', 'data_translated_korean', 'data_translated_arabic', 'data_translated_czech',
                       'data_translated_bengali', 'data_translated_vietnamese', 'data_translated_turkish', 'data_translated_telugu', 'data_translated_farsi'
                       ]

def load(ckpt_dir, peft_model, tokenizer_path):

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path
    )

    tokenizer.pad_token_id = (
        0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )

    if not peft_model:

        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float16
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

def format_example(df, idx, include_answer=True):

    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, k=0):

    prompt = ""
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt