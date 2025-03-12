# Does Unlearning Truly Unlearn? A Black Box Evaluation of LLM Unlearning Methods

Paper link: https://arxiv.org/abs/2411.12103v3

The code is released under the MIT license. 

## Data

The Biology dataset is accessible [here](https://drive.google.com/file/d/1uOS9OwfMC6vOcHnnEGRn71Y8cd3Q0SUf/view?usp=sharing). 

The MMLU dataset is contained in *data/MMLU*. The rephrased MMLU data is contained as subfolders in *data/MMLU-rephrased*.

The WMDP dataset and its rephrased version are not provided as permission is required to access the dataset (see [here](https://github.com/centerforaisafety/wmdp)). 

Details on how the data was rephrased for the rephrasing tests for MMLU and WMDP are contained in the files 'MMLU Prompts.pdf' and 'WMDP Prompts.pdf' respectively.

## Environments

For RMU training, use *environment_RMU.yml* for conda installation. For the rest of the code (LLMU training and all evaluation), use *environment_LLMU.yml*. 

##  Training 

For the hyperparameter values that are not mentioned, the default values in the code are used. Since training is non-deterministic even on setting the seed (see [this](https://pytorch.org/docs/stable/notes/randomness.html)), and both LLMU and RMU training is highly stochastic, using the same hyperparameters/checkpoints as mentioned below is not guaranteed to replicate results, and we recommend trying out different checkpoints or varying the hyperparameters. 

### LLMU

Deepspeed is used for LLMU training due to resource constraints. The same deepspeed config is used throughout all experiments and is located at *training/LLMU/deepspeed_configs/config.json*.

To run LLMU on Biology, run the Python file *training/LLMU/Biology/unlearn_harm.py*. The best hyperparameter settings are as follows:

* Zephyr

    - bad_weight=0.07
    - use_lora
    - drop_last

    Best performance was on checkpoint 5000

* Llama

    - bad_weight=0.09
    - use_lora
    - drop_last

    Best performance was on checkpoint 5000

E.g.: *deepspeed training/LLMU/Biology/unlearn_harm.py --deepspeed --deepspeed_config training/LLMU/deepspeed_configs/config.json --bad_weight 0.07 --model_save_dir LLMU_Biology_Zephyr  --use_lora  --drop_last  --normal_dataset Wikipedia_History_Concepts-in-Physics_Philosophy  --unlearning_dataset Wikipedia_Biology*

To run LLMU on wmdp, run the Python file *training/LLMU/wmdp/unlearn_harm.py*. The best hyperparameter settings are as follows:

* Zephyr

    - bad_weight=0.0036 
    - use_lora
    
    Best performance was on checkpoint 1000

* Llama

    - bad_weight=0.0010 
    - use_lora
    
    Best performance was on checkpoint 1500

E.g. *deepspeed training/LLMU/wmdp/unlearn_harm.py  --deepspeed --deepspeed_config training/LLMU/deepspeed_configs/config.json  --bad_weight 0.0036  --model_save_dir LLMU_wmdp_Zephyr  --wmdp_dataset_path  wmdp/data  --use_lora*


### RMU

To run RMU on biology, run the Python file *training/RMU/Biology/unlearn.py*. The best hyperparameter settings are as follows:


* Zephyr 

    - drop_last
    - alpha=50.0
    - steering_coeffs=10
    - layer_id=15

    Best performance was on checkpoint 200

* Llama

    - drop_last
    - alpha=50.0
    - steering_coeffs=30

    Best performance was on checkpoint 100

E.g. *python  training/RMU/Biology/unlearn.py    --retain_corpus Wikipedia_History_Concepts-in-Physics_Philosophy  --forget_corpus Wikipedia_Biology  --drop_last  --alpha 50  --steering_coeff  10   --layer_id 15  --verbose  --output_dir  RMU_biology_Zephyr*


To run RMU on WMDP, run the Python file *training/RMU/wmdp/unlearn.py*. The best hyperparameter settings are as follows:

* Zephyr 

    The [provided model](https://huggingface.co/cais/Zephyr_RMU) was used 

* LLMU

    - alpha=1200.0,1200.0
    - steering_coeffs=30,30

    Best performance was on checkpoint 150

E.g. *python  training/RMU/wmdp/unlearn.py  --model_name_or_path  meta-llama/Meta-Llama-3-8B-Instruct  --output_dir  RMU_wmdp_Llama   --wmdp_dataset_path  wmdp/data   --alpha 1200,1200  --steering_coeffs  30,30   --verbose*

## Evaluation 

### MMLU & WMDP

Evaluation involves first running a generation script to generate the answers, which are stored in a json file of the form *run_results_ .json*, and then running another script to calculate the results. Use the argument *extra_info* to modify the name of the name of the created file. A breakpoint file of the form *run_breakpoint_ .json* will also be created during generation which will be used by default to resume evaluation in case execution is interrupted. Use the argument *peft_model* in case evaluation is being done on a model trained with LoRA.

For the rephrasing tests, the names of the directories containing the rephrased data (which will be appended to the end of the path provided to the generation files) are in a list in the file *evaluation/common/utils.py*.

#### MMLU

Run the script *evaluation/MMLU/generate_mmlu_responses.py* for generation. For 5-shot prompting, set the value of *ntrain* to 5 and pass the the subject to be used for 5-shot prompting as the *dev_task*. Pass the json file created to *evaluation/MMLU/cal_mmlu_result.py* for evaluation.

To run the rephrasing tests, use the script *evaluation/MMLU/generate_mmlu_responses_rephrasing.py* for generation and *evaluation/MMLU/cal_mmlu_result_rephrasing.py* for evaluation.


#### WMDP

Run the script *evaluation/wmdp/generate_wmdp_responses.py* for generation. For 5-shot prompting, set the value of *ntrain* to 5, pass the path to the MMLU dataset to *MMLU_dir* (since MMLU questions are used for 5-shot prompting), and pass the the subject to be used for 5-shot prompting as the *dev_task*. Pass the json file created to *evaluation/wmdp/cal_wmdp_result.py* for evaluation.

To run the rephrasing tests, use the script *evaluation/wmdp/generate_wmdp_responses_rephrasing.py* for generation and *evaluation/wmdp/cal_wmdp_result_rephrasing.py* for evaluation.

### MT-Bench

Use [this](https://github.com/lm-sys/FastChat) for MT-Bench evaluation.

### Perplexity 

To get the perplexity score, run the *evaluation/perplexity/evaluate.py* script. Set *proportion* to 0.02. The rest of the hyperparameters used are the default ones in the code. 

E.g. *python evaluation/perplexity/evaluate.py --proportion 0.02*

### Openwebtext finetuning 

To retrain on Openwebtext, use the script *evaluation/retraining/retrain.py*.

E.g. *python evaluation/retraining/retrain.py  --model RMU_wmdp_Llama_checkpoint_200  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct  --use-lora  --model_save_dir RMU_wmdp_Llama_checkpoint_200_retrained_model*
