"""The collection of analysis utilities for salad.
"""
import os, sys
import matplotlib.pyplot as plt

def get_loss_row(file: str, 
                 data_type: str, 
                 eval_results: dict, 
                 header: list,
                 key_word_map: dict) -> list:
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
        if key in key_word_map and key_word_map[key] in eval_results and eval_results[key_word_map[key]] is not None:
            _key = key_word_map[key]
            value = eval_results[_key]['avg_loss'][-1]
            if isinstance(value, float):
                if 'nr_'+_key in eval_results:
                    nr = eval_results['nr_'+_key]
                    row.append(f"{value:.4f}({nr/1000000:.2f}M)")
                else:
                    row.append(f"{value:.4f}")
            elif isinstance(value, str):   # Handle case where value is 'N/A'
                row.append(value)
        else:
            row.append('N/A')
    return row

def get_ppl_row(file: str, 
                data_type: str, 
                eval_results: dict, 
                header: list,
                key_word_map: dict) -> list:
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
        if key in key_word_map and key_word_map[key] in eval_results and eval_results[key_word_map[key]] is not None:
            value = eval_results[key_word_map[key]]['ppl']
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            elif isinstance(value, str):   # Handle case where value is 'N/A'
                row.append(value)
        else:
            row.append('N/A')
    return row

def get_acc_row(file: str, 
                data_type: str, 
                eval_results: dict, 
                header: list,
                key_word_map: dict) -> list:
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