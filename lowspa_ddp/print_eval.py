import pickle
import os, sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lowspa_ddp.utils import *

root = get_parent_path(lvl=1)

key_word_map = {
    'baseline': 'baseline',
    'LoR(baseline)': 'baseline_lowrank',
    'X': 'lowspa',
    'X-S': 'lowspa_without_sparsity',
    'LoR(X-S)': 'lowspa_lowrank_without_sparsity',
    'L': 'lowspa_lowrank',
    'LoR(L)': 'lowspa_lowrank_lowrank',
    'L+S': 'lowspa_lowrank_sparsity',
    'LoR(L)+S': 'lowspa_lowrank_lowrank_sparsity'
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
    for key in header:
        if key in key_word_map:
            row.append(f"{eval_results[key_word_map[key]]['loss']:.2f}")
        else:
            row.append('N/A')
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
    for key in header:
        if key in key_word_map:
            row.append(f"{eval_results[key_word_map[key]]['ppl']:.2f}")
        else:
            row.append('N/A')
    return row

def get_acc_row(file: str, data_type: str, eval_results: dict, header: list) -> list:
    """
    Get a row of accuracy statistics for the model.
    Args:
        file: Name of the file containing the statistics.
        data_type: Type of data (e.g., 'train', 'test').
        eval_results: Evaluation results dictionary.
    Returns:
        A list with accuracy statistics.
    """
    row = [file, data_type, 'accuracy']
    for key in header:
        if key in key_word_map:
            row.append(f"{eval_results[key_word_map[key]]['correct']}/{eval_results[key_word_map[key]]['total']}({100.0*eval_results[key_word_map[key]]['accuracy']:.1f}%)")
        else:
            row.append('N/A')
    return row

def get_row(model_type: str, file: str, header: list) -> dict:
    """
    Get a row of statistics for the model from the saved file.
    Args:
        model_type: Type of the model (e.g., 'CNN', 'GPT').
        file: Name of the file containing the statistics.
    Returns:
        A dictionary with statistics for the model.
    """
    path = os.path.join(root, 'data', 'lowspa_ddp', model_type, file)
    with open(os.path.join(path, 'eval_results.pkl'), 'rb') as f:
        stats = pickle.load(f)

    eval_train_results = stats['eval_train_results']
    eval_test_results = stats['eval_test_results']

    row1 = get_loss_row(file, 'train', eval_train_results, header)
    row2 = get_acc_row(file, 'train', eval_train_results, header)
    row3 = get_ppl_row(file, 'train', eval_train_results, header)
    row2[0] = ''
    row2[1] = ''
    row3[0] = ''
    row3[1] = ''


    row4 = get_loss_row(file, 'test', eval_test_results, header)
    row5 = get_acc_row(file, 'test', eval_test_results, header)
    row6 = get_ppl_row(file, 'test', eval_test_results, header)
    row5[0] = ''
    row5[1] = ''
    row6[0] = ''
    row6[1] = ''
    return (row1, row2, row3, row4, row5, row6)

def main(model_type: str, files: list) -> None:
    # headers = [f"model", f"dataset", f"metric", 
    #            f"baseline", f"LoR(baseline)", 
    #            f"X", f"X-S", f"LoR(X-S)", 
    #            f"L", f"LoR(L)", f"L+S", f"LoR(L)+S"]

    headers = [f"model", f"dataset", f"metric", 
               f"baseline", f"LoR(baseline)", f"X", f"X-S", f"LoR(X-S)",
               f"L", f"LoR(L)", f"L+S", f"LoR(L)+S"]
    
    rows = []
    for file in files:
        r1, r2, r3, r4, r5, r6 = get_row(model_type, file, headers[3:])
        rows.append(r1)
        rows.append(r2)
        rows.append(r3)
        rows.append(r4)
        rows.append(r5)
        rows.append(r6)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    model_type = 'GPT'
    # files = ['20250730_215710',
    #          '20250731_084208',
    #          '20250731_130721']

    files = [
            #  '20250731_130721',
            #  '20250731_170700',
            #  '20250731_214246',
             '20250807_091201']
    main(model_type=model_type,
         files=files)