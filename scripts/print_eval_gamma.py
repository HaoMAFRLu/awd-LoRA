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
    rows = []
    row = [file, data_type, 'loss']
    gammas = eval_results.keys()
    sorted_gammas = sorted(gammas, key=lambda x: float(x))
    for gamma in sorted_gammas:
        _row = row + [gamma]
        _eval_results = eval_results[gamma]

        for key in header:
            if key in key_word_map and key_word_map[key] in _eval_results and _eval_results[key_word_map[key]] is not None:
                _key = key_word_map[key]
                value = _eval_results[_key]['avg_loss'][-1]
                if isinstance(value, float):
                    if 'nr_'+_key in _eval_results:
                        nr = _eval_results['nr_'+_key]
                        _row.append(f"{value:.4f}({nr/1000000:.2f}M)")
                    else:
                        _row.append(f"{value:.4f}")
                elif isinstance(value, str):   # Handle case where value is 'N/A'
                    _row.append(value)
            else:
                _row.append('N/A')
        
        rows.append(_row)
    return rows

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
    rows = []
    row = [file, data_type, 'ppl']
    gammas = eval_results.keys()
    sorted_gammas = sorted(gammas, key=lambda x: float(x))
    for gamma in sorted_gammas:
        _row = row + [gamma]
        _eval_results = eval_results[gamma]

        for key in header:
            if key in key_word_map and key_word_map[key] in _eval_results and _eval_results[key_word_map[key]] is not None:
                value = _eval_results[key_word_map[key]]['ppl']
                if isinstance(value, float):
                    _row.append(f"{value:.4f}")
                elif isinstance(value, str):   # Handle case where value is 'N/A'
                    _row.append(value)
            else:
                _row.append('N/A')
        rows.append(_row)
    return rows

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

def get_results(model_type: str, 
                folder: str,
                file: str,
                precision: str) -> dict:
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
    files = os.listdir(path)
    eval_files = [f for f in files if 'eval_results_' + precision in f]
    for file in eval_files:
        with open(os.path.join(path, file), 'rb') as f:
            stats = pickle.load(f)
        # take gamma value from the file name
        gamma = file.rsplit('.', 1)[0].split('_')[-1]            # 提取最后一个
    
        eval_train_results[gamma] = stats['eval_train_results']
        eval_test_results[gamma] = stats['eval_test_results']    

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


def main(model_type: str, 
         folder: str,
         files: list,
         precision: str='bf16') -> None:
    headers = [f"model", f"dataset", f"gamma", f"metric",
               f"X",  
               f"L+S", 
               f'par LoR(L)+S_0', 
               f'par LoR(L)+S_1', 
               f'par LoR(L)+S_2']
    
    rows = []
    for file in files:
        eval_train_results, eval_test_resutls = get_results(model_type, folder, file, precision)
        r1, r2, r3, r4 = get_rows_exp(eval_train_results, eval_test_resutls, file, headers[4:])
        _rows = r1 + r2 + r3 + r4
        for i in range(1, len(_rows)):
            _rows[i][0] = ''
        rows = rows + r1 + r2 + r3 + r4  
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    model_type = 'llama_130m'
    FOLDER = 'baseline'
    files = os.listdir(os.path.join(root, 'data', FOLDER, model_type))
    
    files = [
            '20251210_093825',
            '20251210_093644'    
             ]
    precision = 'fp32'
    main(model_type=model_type,
         folder=FOLDER,
         files=files,
         precision=precision)