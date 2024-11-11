from os import path, sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common.utils import choices

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
