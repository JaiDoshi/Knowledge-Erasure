import json
import argparse
from generation_utils import TASKS, BIOLOGY_TASKS, OTHER_TASKS


choices = ["A", "B", "C", "D"]


data_directory_list = ['data_filler_1', 'data_filler_2', 'data_filler_before_1', 'data_filler_before_2', 
                        'data_filler_before_3',
                        'data_rephrased_conversation_2', 
                       'data_rephrased_poem_1', 'data_rephrased_remove_technical_1', 'data_rephrased_remove_technical_2', 
                       'data_translated_french_2', 'data_translated_german_2', 'data_translated_hindi_2',
                       'data_translated_korean_2', 'data_translated_arabic_2', 'data_translated_czech_2',
                        'data_translated_bengali_2', 'data_translated_vietnamese_2', 'data_translated_turkish_2',
                       'data_translated_telugu_2', 'data_translated_farsi_2',
                       'data_with_variables_2', 'data_filler_before_latin_rephrased_conversation', 'data_filler_before_english_rephrased_conversation',
                       'data_filler_before_hindi_rephrased_conversation', 'data_filler_before_latin_rephrased_poem', 'data_filler_before_english_rephrased_poem',
                       'data_filler_before_hindi_rephrased_poem'
                       ]

data_directory_list = ['data_filler_1', 'data_rephrased_remove_technical_1', 'data_translated_farsi_2'
                       ]


def compute_metric(run_results):


    for directory in data_directory_list:

        print('Printing results from:', directory)

        total_acc = 0
        total_num = 0
        accuracies = {}
        accuracies_answered = {}
        percentage_answered = {}

        for task in TASKS:
            num_answered = 0
            acc = 0
            pred_answers = run_results[directory][task]["pred_answers"]
            gold_answers = run_results[directory][task]["gold_answers"]

            for pred, gold in zip(pred_answers, gold_answers):
                if pred.upper() == gold:
                    acc += 1
                if pred.upper() in choices:
                    num_answered += 1
            accuracies[task] = acc / len(gold_answers)
            accuracies_answered[task] = acc / num_answered if num_answered != 0 else 0
            percentage_answered[task] = num_answered/ len(gold_answers)
            total_acc += acc
            total_num += len(gold_answers)

        accuracies["biology_all"] = sum([accuracies[task] for task in BIOLOGY_TASKS]) / len(
            BIOLOGY_TASKS
        )

        accuracies_answered["biology_all"] = sum([accuracies_answered[task] for task in BIOLOGY_TASKS]) / len(
        BIOLOGY_TASKS
        )
        percentage_answered["biology_all"] = sum([percentage_answered[task] for task in BIOLOGY_TASKS]) / len(
        BIOLOGY_TASKS
        )

        accuracies["other_all"] = sum([accuracies[task] for task in OTHER_TASKS]) / len(
        OTHER_TASKS
        )

        accuracies_answered["other_all"] = sum([accuracies_answered[task] for task in OTHER_TASKS]) / len(
        OTHER_TASKS
        )
        percentage_answered["other_all"] = sum([percentage_answered[task] for task in OTHER_TASKS]) / len(
        OTHER_TASKS
        )

        accuracies["all_subjects"] = sum([accuracies[task] for task in TASKS]) / len(
            TASKS
        )
        accuracies_answered["all_subjects"] = sum([accuracies_answered[task] for task in TASKS]) / len(
            TASKS
        )
        percentage_answered["all_subjects"] = sum([percentage_answered[task] for task in TASKS]) / len(
            TASKS
        )

        print("ACC-biology: %.4f" % accuracies["biology_all"])
        print("ACC-biology-answered: %.4f" % accuracies_answered["biology_all"])
        print("Percentage-biology-answered: %.4f" % percentage_answered["biology_all"])
        print("-----------------")
        print("ACC-other: %.4f" % accuracies["other_all"])
        print("ACC-other-answered: %.4f" % accuracies_answered["other_all"])
        print("Percentage-other-answered: %.4f" % percentage_answered["other_all"])
        print("-----------------")
        print("ACC-all_subjects: %.4f" % accuracies["all_subjects"])
        print("ACC-all_subjects-answered: %.4f" % accuracies_answered["all_subjects"])
        print("Percentage-all_subjects-answered: %.4f" % percentage_answered["all_subjects"])
        print("-----------------")


def main(args):

    run_results = json.load(open(args.file_name, "r"))
    compute_metric(run_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=True)
    args = parser.parse_args()

    main(args)