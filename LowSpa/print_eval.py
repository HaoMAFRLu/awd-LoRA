"""This script is used to pretrain a CNN for 
late
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
import pickle
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn import CNN
from dataloaders.dataloader import get_mnist
from utils.general import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = get_parent_path(lvl=1)

def main(model_name=['CNN.pth']):
    data_list = []
    for model in model_name:
        path_file = os.path.join(root, 'data', 'LowSpa', model + '_eval')
        with open(path_file, 'rb') as file:
            data = pickle.load(file)
        data_list.append(data)

    for data in data_list:
        print(f"Evaluating model: {data['model_name']}")
        if data['model_name'] == 'normal':
            # print all the normal information
            print(f"Model: {data['model_name']}\n"
                  f"Train Loss (X): {data['train_loss']:.4f} | Acc: {100. * data['train_accuracy']:.4f}%\n"
                  f"Test Loss (X): {data['test_loss']:.4f} | Acc: {100. * data['test_accuracy']:.4f}%\n"
                  f"Train Loss (X90%): {data['train_loss1']:.4f} | Acc: {100. * data['train_accuracy1']:.4f}%\n"
                  f"Test Loss (X90%): {data['test_loss1']:.4f} | Acc: {100. * data['test_accuracy1']:.4f}%\n"
                  f"Layer 1: {data['idx1']}/{data['s1']}\n"
                  f"Layer 2: {data['idx2']}/{data['s2']}\n")
        else:
            print(f"Model: {data['model_name']}\n"
                    f"Train Loss (X): {data['train_loss']:.4f} | Acc: {100. * data['train_accuracy']:.4f}%\n"
                    f"Test Loss (X): {data['test_loss']:.4f} | Acc: {100. * data['test_accuracy']:.4f}%\n"
                    f"Train Loss (X90%): {data['train_loss6']:.4f} | Acc: {100. * data['train_accuracy6']:.4f}%\n"
                    f"Test Loss (X90%): {data['test_loss6']:.4f} | Acc: {100. * data['test_accuracy6']:.4f}%\n"
                    f"Train Loss (X-S): {data['train_loss1']:.4f} | Acc: {100. * data['train_accuracy1']:.4f}%\n"
                    f"Test Loss (X-S): {data['test_loss1']:.4f} | Acc: {100. * data['test_accuracy1']:.4f}%\n"
                    f"Train Loss (L+S): {data['train_loss2']:.4f} | Acc: {100. * data['train_accuracy2']:.4f}%\n"
                    f"Test Loss (L+S): {data['test_loss2']:.4f} | Acc: {100. * data['test_accuracy2']:.4f}%\n"
                    f"Train Loss (L): {data['train_loss3']:.4f} | Acc: {100. * data['train_accuracy3']:.4f}%\n"
                    f"Test Loss (L): {data['test_loss3']:.4f} | Acc: {100. * data['test_accuracy3']:.4f}%\n"
                    f"Train Loss (L90%): {data['train_loss4']:.4f} | Acc: {100. * data['train_accuracy4']:.4f}%\n"
                    f"Test Loss (L90%): {data['test_loss4']:.4f} | Acc: {100. * data['test_accuracy4']:.4f}%\n"
                    f"Train Loss (L90%+S): {data['train_loss5']:.4f} | Acc: {100. * data['train_accuracy5']:.4f}%\n"
                    f"Test Loss (L90%+S): {data['test_loss5']:.4f} | Acc: {100. * data['test_accuracy5']:.4f}%\n"
                    f"========== Layer 1 ==============\n"
                    f"Rank (X/X-S/full): {data['_s1']}/{data['s1']}/{data['full_s1']}\n"
                    f"Loss (X = L + S): {data['l1']:.6f}\n"
                    f"Non-zero elements: {data['n1']}/{data['total_elements1']} | {100. * data['n1']/data['total_elements1']:.4f}%\n"
                    f"========== Layer 2 ==============\n"
                    f"Rank (X/X-S/full): {data['_s2']}/{data['s2']}/{data['full_s2']}\n"
                    f"Loss (X = L + S): {data['l2']:.6f}\n"
                    f"Non-zero elements: {data['n2']}/{data['total_elements2']} | {100. * data['n2']/data['total_elements2']:.4f}%\n")

if __name__ == '__main__':
    model_name = ['normal',
                #   '512_50_0.008_[0.1, 0.1]_[0.008, 0.02]_[0.05, 0.05]',
                #   '512_50_0.008_[0.05, 0.05]_[0.008, 0.02]_[0.05, 0.05]',
                #   '512_50_0.008_[0.1, 0.1]_[0.008, 0.02]_[0.01, 0.01]',
                #   '512_50_0.008_[0.1, 0.1]_[0.01, 0.022]_[0.005, 0.005]',
                #   '512_50_0.008_[0.15, 0.1]_[0.012, 0.02]_[0.002, 0.005]',
                #   '512_50_0.008_[0.1, 0.1]_[0.0065, 0.02]_[0.002, 0.005]',
                #   '512_50_0.008_[0.1, 0.1]_[0.0075, 0.02]_[0.02, 0.05]',
                #   '512_50_0.008_[0.1, 0.1]_[0.008, 0.02]_[0.2, 0.05]',
                #   '512_50_0.008_[0.1, 0.15]_[0.008, 0.028]_[0.05, 0.05]',
                #   '512_50_0.008_[0.1, 0.5]_[0.008, 0.1]_[0.05, 0.05]_100_0.001',
                #   '512_50_0.008_[0.12, 0.1]_[0.007, 0.02]_[0.01, 0.01]_100_0.001',
                #   '512_50_0.008_[0.12, 0.1]_[0.008, 0.02]_[0.01, 0.01]_10_0.001',
                #   '512_50_0.008_[0.12, 0.1]_[0.008, 0.02]_[0.01, 0.01]_1_0.001',
                #   '512_100_0.008_[0.12, 0.1]_[0.008, 0.02]_[0.01, 0.01]_1_0.001',
                #   '512_50_0.008_[0.12, 0.1]_[0.008, 0.02]_[0.01, 0.01]_1_0.001',
                #   '512_50_0.008_[0.12, 0.1]_[0.0075, 0.025]_[0.01, 0.01]_1_0.001_5_0.7',
                #   '512_50_0.008_[0.12, 0.1]_[0.0072, 0.02]_[0.01, 0.01]_1_0.001_5_0.7',
                #   '512_100_0.008_[0.12, 0.1]_[0.008, 0.02]_[0.01, 0.01]_1_0.001_5_0.7',
                #   '512_100_0.008_[0.0012, 0.01]_[8e-05, 0.002]_[0.0001, 0.001]_1_0.001_5_0.7',  
                  '512_100_0.008_[0.0001, 0.0005]_[8e-06, 0.0001]_[0.0001, 0.001]_1_0.001_5_0.7'] 
    
    main(model_name=model_name)