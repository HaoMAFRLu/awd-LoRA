import pickle
import os, sys
import torch
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import *

root = get_parent_path(lvl=1)

key_word_map = {
    'X': 'X',
    'X-S': 'X_without_S',
    'LoR(X-S)': 'lowrank_X_without_S',
    'L': 'L',
    'LoR(L)': 'lowrank_L',
    'L+S': 'L_with_S',
    'par L+S': 'par_L_with_S',
    'LoR(L)+S': 'lowrank_L_with_S',
    'spe LoR(L)+S': 'lowrank_L_with_S_specify',
    'par LoR(L)+S': 'par_lowrank_L_with_S',
    'par LoR(L)+S_0': 'par_lowrank_L_with_S_0',
    'par LoR(L)+S_1': 'par_lowrank_L_with_S_1',
    'par LoR(L)+S_2': 'par_lowrank_L_with_S_2',
    'par LoR(L)+S_3': 'par_lowrank_L_with_S_3',
}

def get_loss_row(file: str, data_type: str, eval_results: dict, header: list) -> list:
    """
    Get a row of loss statistics for the model.
    Args:
        file: Name of the file containing the statistics.
        eval_results: Evaluation results dictionary.
    Returns:
        A list with loss statistics.
    """
    row = [file, data_type, 'loss']
    value = eval_results['avg_loss'][-1]
    if isinstance(value, float):
        row.append(f"{value:.4f}")
    elif isinstance(value, str):   # Handle case where value is 'N/A'
        row.append(value)
    return row

def get_ppl_row(file: str, data_type: str, eval_results: dict, header: list) -> list:
    """
    Get a row of perplexity statistics for the model.
    Args:
        file: Name of the file containing the statistics.
        data_type: Type of data (e.g., 'train', 'test').
        eval_results: Evaluation results dictionary.
    Returns:
        A list with perplexity statistics.
    """
    row = [file, data_type, 'ppl']
    value = eval_results['ppl']
    if isinstance(value, float):
        row.append(f"{value:.4f}")
    elif isinstance(value, str):   # Handle case where value is 'N/A'
        row.append(value)
    return row

def get_results(model_type: str, 
                folder: str,
                file: str) -> dict:
    """
    Get evaluation results for the model.
    Args:
        model_type: Type of the model (e.g., 'CNN', 'GPT').
        file: Name of the file containing the statistics.
        data_type: Type of data (e.g., 'train', 'test').
    Returns:
        A dictionary with evaluation results.
    """
    eval_train_results = {}
    eval_test_results = {}

    path = os.path.join(root, 'data', folder, model_type, file)
    
    with open(os.path.join(path, 'eval_results.pkl'), 'rb') as f:
        stats = pickle.load(f)
    # take gamma value from the file name

    eval_train_results = stats['eval_train_results']
    eval_test_results = stats['eval_test_results']    

    return eval_train_results, eval_test_results

def get_rows_exp(eval_train_results, eval_test_results, file: str, header: list) -> dict:
    """
    Get the rows of statistics for the model from the saved file.
    """
    row1 = get_loss_row(file, 'train', eval_train_results, header)
    row2 = get_ppl_row(file, 'train', eval_train_results, header)
    # row2[0] = ''
    # row2[1] = ''


    row3 = get_loss_row(file, 'test', eval_test_results, header)
    row4 = get_ppl_row(file, 'test', eval_test_results, header)
    # row3[0] = ''
    # row4[0] = ''
    # row4[1] = ''
    return (row1, row2, row3, row4)


def main(path_part: dict) -> None:
    headers = [f"model", f"dataset",  f"metric", f"X"]
    model_type = path_part['model_type']
    folder = path_part['folder']
    file = path_part['file']

    rows = []
    for file in files:
        eval_train_results, eval_test_resutls = get_results(model_type, folder, file)
        r1, r2, r3, r4 = get_rows_exp(eval_train_results, eval_test_resutls, file, headers[3:])
        rows.append(r1)
        rows.append(r2)
        rows.append(r3)
        rows.append(r4)
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    MODEL_TYPES = [
            'llama_60m'
        ]
    FOLDERS = ['vanilla']
    
    files = [
            '20251209_220046'
             ]

    path_part = determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                    FOLDERS=FOLDERS,
                                    file=files[0])

    main(path_part)