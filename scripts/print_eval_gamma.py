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


def main(path_parts: list,
         precision: str='bf16') -> None:
    
    headers = [f"model", f"dataset", f"gamma", f"metric",
               f"X",  
               f"L+S", 
               f'par LoR(L)+S_0', 
               f'par LoR(L)+S_1', 
               f'par LoR(L)+S_2']
    
    rows = []
    for path_part in path_parts:

        model_type = path_part['model_type']
        folder = path_part['folder']
        file = path_part['file']
        
        eval_train_results, eval_test_resutls = get_results(model_type, folder, file, precision)
        r1, r2, r3, r4 = get_rows_exp(eval_train_results, eval_test_resutls, file, headers[4:])
        _rows = r1 + r2 + r3 + r4
        for i in range(1, len(_rows)):
            _rows[i][0] = ''
        rows = rows + r1 + r2 + r3 + r4  
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    MODEL_TYPES = [
                   'llama_9m',
                   'llama_60m',
                   'llama_130m',
                   'llama_350m',
                   'llama_1b'
                ]
    
    FOLDERS = [
        'baseline_fp32',
        'head_fp32',
        'head_bf16',
        'baseline', 
        'incl_embedding'
    ]

    files = [
        # '20251222_124113',
        # '20251222_123930',
        # '20251222_123813',
        # '20251222_115648',
        # '20251222_115509',
        # '20251213_111828',
        # '20251213_102716',
        # '20251210_181001',
        # '20251223_104311',
        # '20251223_104640',
        # '20251224_112407',
        # '20251224_112550',
        # '20251224_112728',
        # '20251224_114115',
        # '20251224_114122',
        # '20251225_141553',
        # '20251225_141411',
        # '20251225_141246',
        # '20251226_155538',
        # '20251226_142518',
        # '20251226_142353',
        # '20251227_220032',
        # '20251227_215821',
        # '20251227_215309',
        # '20251227_215116',
        # '20251227_222332',
        # '20251227_222811',
        # '20251228_114512',
        # '20251228_201955',
        # '20251229_124734',
        # '20251227_122124',
        # '20251227_122303',
        # '20251227_122450',
        # '20251231_124221',
        # '20260101_025548',
        # '20260101_023254',
        # '20260101_023116',
        # '20260102_102653',
        # '20260102_111434',
        # '20260102_234230',
        # '20260102_233510',
        # '20260105_090228',
        # '20251209_104454',  # for quick test
        # '20251204_135646',  # 60m, baseline fp32
        # '20251204_152747',  # 60m, head fp32
        # '20251227_222811',  # 60m, head bf16
        # '20251202_164626',  # 130m, baseline fp32
        # '20251203_144749',  # 130m, head fp32
        # '20251227_222332',  # 130m, head bf16
        # '20251203_102315',  # 350m, baseline fp32
        # '20251204_134313',  # 350m, head fp32
        # '20260102_233510',  # 350m, head bf16
        # '20251130_125959',  # 1b, baseline fp32
        '20260105_090228',
    ]


    precision = 'bf16'

    path_parts = []
    for file in files:
        path_parts.append(determine_path_part(MODEL_TYPES=MODEL_TYPES,
                                              FOLDERS=FOLDERS,
                                              file=file))

    main(path_parts,
         precision=precision)