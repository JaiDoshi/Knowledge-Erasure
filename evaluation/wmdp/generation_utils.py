choices = ["A", "B", "C", "D"]

TASKS = ['bio_questions', 'cyber_questions']


def format_wmdp_example(question, include_answer=False):
        
    prompt = question['question']
    k = len(question['choices'])
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], question['choices'][j])
    prompt += "\nAnswer:"
    if include_answer:
            prompt += " {}\n\n".format(chr(question['answer'] + 65))
    return prompt

def format_example(df, idx, include_answer=True):
        
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, k=-1):

    if k == -1:
        k = train_df.shape[0]

    prompt = ""
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt




