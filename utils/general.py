"""Collection of useful functions.
"""
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import torch
from tqdm import tqdm
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn import CNN


def mkdir(path: Path) -> None:
    """Check if the folder exists and create it
    if it does not exist.
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def get_parent_path(lvl: int=0) -> Path:
    """Get the lvl-th parent path as root path.
    Return current file path when lvl is zero.
    Must be called under the same folder.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if lvl > 0:
        for _ in range(lvl):
            path = os.path.abspath(os.path.join(path, os.pardir))
    return path

def get_folder_name() -> str:
    """Return a folder name based on the current time.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S") 

def get_pretrained(root,
                   device):
    model = CNN().to(device)
    path_file = os.path.join(root, 'models', 'pretrained', 'CNN.pth')
    state_dict = torch.load(path_file, map_location=device)
    model.load_state_dict(state_dict)
    return model.eval()

def rpca_admm(W, la=0.1, mu=0.1, max_iter=500, tol=1e-6):
    """
    Robust PCA using ADMM.
    Args:
        W: Input matrix to decompose.
        la: Regularization parameter for the low-rank component.
        mu: Regularization parameter for the sparse component.
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
    Returns:
        L: Low-rank component.
        S: Sparse component.
    """
    m, n = W.shape
    L = np.zeros((m, n))
    S = np.zeros((m, n))
    Y = np.zeros((m, n))
    
    for it in range(max_iter):
        # Update L
        U, s, Vt = np.linalg.svd(W - S + Y / mu, full_matrices=True)
        Sigma = np.zeros((m, n))
        for i in range(len(s)):
            Sigma[i, i] = s[i]
        L = U@soft_threshold(Sigma, 1/mu)@Vt
        S = soft_threshold(W - L + Y/mu, la/mu)
        Y = Y + mu * (W - L - S)
        
        # Check convergence
        if np.linalg.norm(W - L - S, 'fro') < tol:
            break
    
    print(f"None zero elements in S: {np.count_nonzero(S)}/{m*n}")
    print(f'Loss is: {np.linalg.norm(W - L - S, "fro"):.7f}')
    print(f"ADMM converged in {it+1} iterations.")
    return L, S

def soft_threshold(x: np.ndarray, threshold: float):   
    """
    Apply soft thresholding to the input array.
    Args:
        x: Input array.
        threshold: Threshold value.
    Returns:
        Soft-thresholded array.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def rpca_admm_conv2d(W, la=0.1, mu=0.1, max_iter=500, tol=1e-6):
    """
    Robust PCA for 2D convolutional layers using ADMM.
    Args:
        W: Input matrix to decompose.
        la: Regularization parameter for the low-rank component.
        mu: Regularization parameter for the sparse component.
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
    Returns:
        L: Low-rank component.
        S: Sparse component.
    """
    cout, cin, h, w = W.shape
    W_flat = W.reshape(cout, -1)
    L_flat, S_flat = rpca_admm(W_flat, la, mu, max_iter, tol)
    return L_flat.reshape(cout, cin, h, w), S_flat.reshape(cout, cin, h, w)

def evaluate_per_class(model, dataloader, device, nr_class):
    """
    Evaluate the model on the dataloader and return accuracy per class.
    Args:
        model: The model to evaluate.
        dataloader: DataLoader containing the dataset.
        device: Device to run the evaluation on.
    Returns:
        train_acc: Overall accuracy on the dataset.
        train_acc_per_class: Accuracy per class.
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * nr_class
    class_total = [0] * nr_class
    
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                class_correct[label] += (predicted[i] == label).item()
    
    train_acc = correct / total * 100
    train_acc_per_class = [class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0 for i in range(nr_class)]
    
    return train_acc, train_acc_per_class