"""Run the LM Harness evaluation for a given model and a given set of tasks. 
"""
import sys, os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from datetime import datetime
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

hf_login_once()  # To avoid the error 429 when downloading the model from HuggingFace Hub for the first time.

root = get_parent_path(lvl=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16
batch_size = 16
num_fewshot = 0
# TASKS = ['piqa', 'boolq']
TASKS = [
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "copa",
    "mmlu",
    "hellaswag",
    "gsm8k",
    "truthfulqa",
]

def main(MODEL_TYPE: str,
         FOLDER: str,
         file: str,):
    
    MODEL_PATH = os.path.join(root, 'data', FOLDER, MODEL_TYPE, file, 'model_resave')

    model = HFLM(
        pretrained=MODEL_PATH,
        dtype=dtype,
        device=str(device),
        batch_size=batch_size,
    )

    results = evaluator.simple_evaluate(
    model=model,
    tasks=TASKS,
    num_fewshot=num_fewshot,
    )

    OUTPUT_DIR = os.path.join(root, 'data', FOLDER, MODEL_TYPE, file, 'lm_harness_eval_results')
    mkdir(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, f"results")

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_path}")


if __name__ == '__main__':

    MODEL_TYPES = [
                   'llama_9m',
                   'llama_60m',
                   'llama_130m',
                   'llama_350m',
                   'llama_1b'
                ]
    
    FOLDERS = [
        'vanilla',
        'baseline', 
        'incl_embedding',
        'head',
        'baseline_fp32',
        'head_fp32',
        'head_bf16',
        'vanilla_bf16',
    ]

    files = [
        # '20251209_104454',
        '20251130_125959',
    ]

    for file in files:
        path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                        FOLDERS=FOLDERS,
                                        file=file)

        MODEL_TYPE = path_part['model_type']
        FOLDER = path_part['folder']
       
        main(MODEL_TYPE=MODEL_TYPE, 
             FOLDER=FOLDER, 
             file=file)
    