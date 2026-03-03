"""Read the results from lm eval.
"""
"""Run the LM Harness evaluation for a given model and a given set of tasks. 
"""
import sys, os
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *


root = get_parent_path(lvl=1)

def main(MODEL_TYPE: str,
         FOLDER: str,
         file: str,):
    
    path_folder = os.path.join(root, 'data', FOLDER, MODEL_TYPE, file, 'lm_harness_eval_results')
    files = os.listdir(path_folder)
    result_files = [f for f in files if f.endswith('results')]

    for file in result_files:
        with open(os.path.join(path_folder, file), "rb") as f:
            results = pickle.load(f)
        
        print(f"Results from {file}:")
        for task in results['results']:
            if task != 'gsm8k' and task != 'truthfulqa_gen':
                print(f"  {task}: {results['results'][task]['acc,none']:.4f}")
        print()

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
    