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
    'par LoR(L)+S': 'par_lowrank_L_with_S'
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
            value = eval_results[key_word_map[key]]['avg_loss'][-1]
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            elif isinstance(value, str):   # Handle case where value is 'N/A'
                row.append(value)
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
            value = eval_results[key_word_map[key]]['ppl']
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            elif isinstance(value, str):   # Handle case where value is 'N/A'
                row.append(value)
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

def get_results(model_type: str, file: str, data_type: str) -> dict:
    """
    Get evaluation results for the model.
    Args:
        model_type: Type of the model (e.g., 'CNN', 'GPT').
        file: Name of the file containing the statistics.
        data_type: Type of data (e.g., 'train', 'test').
    Returns:
        A dictionary with evaluation results.
    """
    path = os.path.join(root, 'data', 'salad', model_type, file)
    with open(os.path.join(path, 'eval_results.pkl'), 'rb') as f:
        stats = pickle.load(f)

    eval_train_results = stats['eval_train_results']
    eval_test_results = stats['eval_test_results']

    return eval_train_results, eval_test_results

def get_row(eval_train_results, eval_test_results, file: str, header: list) -> dict:
    """
    Get a row of statistics for the model from the saved file.
    Args:
        model_type: Type of the model (e.g., 'CNN', 'GPT').
        file: Name of the file containing the statistics.
    Returns:
        A dictionary with statistics for the model.
    """
    row1 = get_loss_row(file, 'train', eval_train_results, header)
    row2 = get_ppl_row(file, 'train', eval_train_results, header)
    row2[0] = ''
    row2[1] = ''


    row3 = get_loss_row(file, 'test', eval_test_results, header)
    row4 = get_ppl_row(file, 'test', eval_test_results, header)
    row3[0] = ''
    row4[0] = ''
    row4[1] = ''
    return (row1, row2, row3, row4)

def plot_loss(eval_train_results, eval_test_results, header: list, pth_fig: str) -> None:
    nr_heads = len(header)
    fig, ax = plt.subplots(nr_heads, 1, figsize=(10, 6*nr_heads))
    for i, key in enumerate(header):
        if key in key_word_map and isinstance(eval_train_results[key_word_map[key]]['avg_loss'][-1], float):
            set_axes_format(ax[i], r'Iterations', r'Loss')
            ax[i].plot(eval_train_results[key_word_map[key]]['avg_loss'], label='Train Loss')
            ax[i].plot(eval_test_results[key_word_map[key]]['avg_loss'], label='Test Loss')
            ax[i].set_title(f"{key} Loss")
            ax[i].set_xlabel('Iterations')
            ax[i].set_ylabel('Loss')
            ax[i].legend()
            ax[i].grid(True)
        else:
            ax[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(pth_fig, 'loss.png'))


def main(model_type: str, files: list) -> None:
    headers = [f"model", f"dataset", f"metric", 
               f"X", f"X-S", f"LoR(X-S)",
               f"L", f"LoR(L)", f"L+S",
               f"LoR(L)+S", f"par LoR(L)+S"]
    
    rows = []
    for file in files:
        eval_train_results, eval_test_resutls = get_results(model_type, file, 'train')
        r1, r2, r3, r4 = get_row(eval_train_results, eval_test_resutls, file, headers[3:])
        rows.append(r1)
        rows.append(r2)
        rows.append(r3)
        rows.append(r4)

        pth_fig = os.path.join(root, 'data', 'salad', model_type, file, 'figures', 'eval')
        mkdir(pth_fig)
        plot_loss(eval_train_results, eval_test_resutls, headers[3:], pth_fig)
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    model_type = 'llama_60m'
    files = ['20250814_150324',
             '20250816_004617',
             '20250816_205604',
             '20250817_140155',]
    main(model_type=model_type,
         files=files)