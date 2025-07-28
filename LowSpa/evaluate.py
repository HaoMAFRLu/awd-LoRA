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
from datasets.dataloader import get_mnist
from utils.general import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = get_parent_path(lvl=1)

def evaluate(model, 
             test_loader,
             loss_fn):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, y in test_loader:
            data = data.to(device)
            y = y.to(device)
            output = model(data)
            loss = loss_fn(output, y)

            test_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1) 
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate_normal_model(model_name, train_loader, test_loader):
    """
    Evaluate the normal model on the train and test set.
    Args:
        model_name: Name of the model to evaluate.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
    Returns:
        Dictionary with evaluation results.
    """
    # build the model
    model = CNN().to(device)
    model_path = os.path.join(root, 'models', 'pretrained', 'CNN.pth')
    # load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss, train_accuracy = evaluate(model, train_loader, loss_fn)
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)

    X1 = model.fc1.weight.data
    X2 = model.fc2.weight.data

    U1, s1, Vt1 = torch.linalg.svd(X1, full_matrices=False)
    U2, s2, Vt2 = torch.linalg.svd(X2, full_matrices=False)

    energy1 = torch.cumsum(s1**2, dim=0) / torch.sum(s1**2)
    energy2 = torch.cumsum(s2**2, dim=0) / torch.sum(s2**2)
    idx1 = torch.where(energy1 >= 0.9)[0][0]
    idx2 = torch.where(energy2 >= 0.9)[0][0]

    model.fc1.weight.data = U1[:, :idx1+1] @ torch.diag(s1[:idx1+1]) @ Vt1[:idx1+1, :]
    model.fc2.weight.data = U2[:, :idx2+1] @ torch.diag(s2[:idx2+1]) @ Vt2[:idx2+1, :] 
    train_loss1, train_accuracy1 = evaluate(model, train_loader, loss_fn)
    test_loss1, test_accuracy1 = evaluate(model, test_loader, loss_fn)

    data = {
        'model_name': model_name,
        'train_loss': train_loss,
        'train_loss1': train_loss1,
        'train_accuracy': train_accuracy,
        'train_accuracy1': train_accuracy1,
        'test_loss': test_loss,
        'test_loss1': test_loss1,
        'test_accuracy': test_accuracy,
        'test_accuracy1': test_accuracy1,
        's1': len(s1),
        's2': len(s2),
        'idx1': idx1 + 1,
        'idx2': idx2 + 1,
    }
    return data

def get_matrix(file_path): 
    """
    Get the matrices from the file.
    Args:
        file_path: Path to the file containing the matrices.
    Returns:
        L1, L2, S1, S2: Matrices from the file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    L1 = torch.tensor(data['L1']).to(device)
    S1 = torch.tensor(data['S1']).to(device)
    L2 = torch.tensor(data['L2']).to(device)       
    S2 = torch.tensor(data['S2']).to(device)
    return L1, L2, S1, S2

def get_layer_info(X, L, S):
    """
    Get the layer information from the model.
    Args:
        X: Weight matrix of the layer.
        L: Low-rank matrix.
        S: Sparse matrix.
    Returns:
        X1, s1, _s1, l1, n1, idx1: Layer information.
    """
    U, s, Vt = torch.linalg.svd(X - S, full_matrices=False)
    _U, _s, _Vt = torch.linalg.svd(X, full_matrices=False)

    l = torch.linalg.norm(X - L - S, 'fro')
    n = torch.count_nonzero(S)

    # find 90% energy of singular values
    energy = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
    idx = torch.where(energy >= 0.9)[0][0]

    _energy = torch.cumsum(_s**2, dim=0) / torch.sum(_s**2)
    _idx = torch.where(_energy >= 0.9)[0][0]
    return X, s, _s, l, n, idx, _idx

def get_90_percent_low_rank(L):
    """
    Get the low-rank matrix with 90% energy.
    Args:
        L: Low-rank matrix.
    Returns:
        L_low: Low-rank matrix with 90% energy.
    """
    U, s, Vt = torch.linalg.svd(L, full_matrices=False)
    energy = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
    idx = torch.where(energy >= 0.9)[0][0]
    L_low = U[:, :idx+1] @ torch.diag(s[:idx+1]) @ Vt[:idx+1, :]
    return L_low

def evaluate_lowspa_model(model_name, train_loader, test_loader):
    """
    Evaluate the LowSpa model on the train and test set.
    Args:
        model_name: Name of the model to evaluate.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
    Returns:
        Dictionary with evaluation results.
    """
    # build the model
    model = CNN().to(device)
    model_path = os.path.join(root, 'data', 'LowSpa', model_name + '.pth')
    # load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loss, train_accuracy = evaluate(model, train_loader, loss_fn)
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)

    file_path = os.path.join(root, 'data', 'LowSpa', model_name+'_matrix')    
    L1, L2, S1, S2 = get_matrix(file_path)
    
    X1, s1, _s1, l1, n1, idx1, _idx1 = get_layer_info(model.fc1.weight.data, L1, S1)
    X2, s2, _s2, l2, n2, idx2, _idx2 = get_layer_info(model.fc2.weight.data, L2, S2)

    # evaluate the complete model
    train_loss, train_accuracy = evaluate(model, train_loader, loss_fn)
    test_loss, test_accuracy = evaluate(model, test_loader, loss_fn)

    model.fc1.weight.data = X1 - S1
    model.fc2.weight.data = X2 - S2

    train_loss1, train_accuracy1 = evaluate(model, train_loader, loss_fn)
    test_loss1, test_accuracy1 = evaluate(model, test_loader, loss_fn)

    model.fc1.weight.data = L1 + S1
    model.fc2.weight.data = L2 + S2

    train_loss2, train_accuracy2 = evaluate(model, train_loader, loss_fn)
    test_loss2, test_accuracy2 = evaluate(model, test_loader, loss_fn)

    model.fc1.weight.data = L1
    model.fc2.weight.data = L2

    train_loss3, train_accuracy3 = evaluate(model, train_loader, loss_fn)
    test_loss3, test_accuracy3 = evaluate(model, test_loader, loss_fn)

    L1_low = get_90_percent_low_rank(L1)
    L2_low = get_90_percent_low_rank(L2)
    model.fc1.weight.data = L1_low
    model.fc2.weight.data = L2_low
    train_loss4, train_accuracy4 = evaluate(model, train_loader, loss_fn)
    test_loss4, test_accuracy4 = evaluate(model, test_loader, loss_fn)

    model.fc1.weight.data = L1_low + S1
    model.fc2.weight.data = L2_low + S2
    train_loss5, train_accuracy5 = evaluate(model, train_loader, loss_fn)
    test_loss5, test_accuracy5 = evaluate(model, test_loader, loss_fn)

    X1_low = get_90_percent_low_rank(X1)
    X2_low = get_90_percent_low_rank(X2)    
    model.fc1.weight.data = X1_low
    model.fc2.weight.data = X2_low
    train_loss6, train_accuracy6 = evaluate(model, train_loader, loss_fn)
    test_loss6, test_accuracy6 = evaluate(model, test_loader, loss_fn)

    data = {
        'model_name': model_name,
        'total_elements1': X1.numel(),
        'total_elements2': X2.numel(),
        'train_loss': train_loss,
        'train_loss1': train_loss1,
        'train_loss2': train_loss2,
        'train_loss3': train_loss3,
        'train_loss4': train_loss4,
        'train_loss5': train_loss5,
        'train_loss6': train_loss6,
        'train_accuracy': train_accuracy,
        'train_accuracy1': train_accuracy1,
        'train_accuracy2': train_accuracy2,
        'train_accuracy3': train_accuracy3,
        'train_accuracy4': train_accuracy4,
        'train_accuracy5': train_accuracy5,
        'train_accuracy6': train_accuracy6,
        'test_loss': test_loss,
        'test_loss1': test_loss1,
        'test_loss2': test_loss2,
        'test_loss3': test_loss3,
        'test_loss4': test_loss4,
        'test_loss5': test_loss5,
        'test_loss6': test_loss6,
        'test_accuracy': test_accuracy,
        'test_accuracy1': test_accuracy1,
        'test_accuracy2': test_accuracy2,
        'test_accuracy3': test_accuracy3,
        'test_accuracy4': test_accuracy4,
        'test_accuracy5': test_accuracy5,
        'test_accuracy6': test_accuracy6,
        'full_s1': len(s1),
        'full_s2': len(s2),
        '_s1': _idx1 + 1,
        '_s2': _idx2 + 1,
        's1': idx1 + 1,
        's2': idx2 + 1,
        'l1': l1,
        'l2': l2,
        'n1': n1,
        'n2': n2
    }

    return data

def evaluate_model(model_name, train_loader, test_loader):  
    if model_name == 'normal':
        data = evaluate_normal_model(model_name, train_loader, test_loader)
    else:
        data = evaluate_lowspa_model(model_name, train_loader, test_loader)
    return data


def main(batch_size,
         model_name=['CNN.pth']):
    # get the dataloader
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    # import models
    data_list = []
    for model in model_name:
        print(f"Evaluating model: {model}")
        data_list.append(evaluate_model(model, train_loader, test_loader))

    for data in data_list:
        model = data['model_name']
        path_file = os.path.join(root, 'data', 'LowSpa', model + '_eval')
        with open(path_file, 'wb') as file:
            pickle.dump(data, file)
    
if __name__ == '__main__':
    model_name = ['512_100_0.008_[0.0001, 0.0005]_[8e-06, 0.0001]_[0.0001, 0.001]_1_0.001_5_0.7'] 
    
    main(batch_size=512,
         model_name=model_name)