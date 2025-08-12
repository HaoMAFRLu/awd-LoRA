"""This script is used to pretrain a CNN for 
late
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
import pickle
from tqdm import tqdm
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn import CNN
from dataloaders.dataloader import get_mnist
from utils.general import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = get_parent_path(lvl=1)

def get_loss_term(X, L, S, Y, mu):
    """
    Compute the loss term for the model.
    Args:
        X: Input data.
        L: Low-rank component.
        S: Sparse component.
        Y: Dual variable.
    Returns:
        Loss term value.
    """
    return mu/2 * torch.norm(X - L - S + Y/mu, p='fro') ** 2
    # return mu/2 * torch.norm(X - L, p='fro') ** 2

def soft_threshold(x: torch.Tensor, threshold: float):
    """
    Apply soft thresholding to the input tensor.
    Args:
        x: Input tensor.
        threshold: Threshold value.
    Returns:
        Soft-thresholded tensor.
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0, device=x.device))

def PRCA(X, L, S, Y, alpha, beta, rho, iter_max=100, tol=1e-3):
    """
    Perform the Principal Component Analysis (PCA) using Robust PCA.
    Args:
        X: Input data.
        L: Low-rank component.
        S: Sparse component.
        Y: Dual variable.
        mu: Regularization parameter for the dual variable.
        la: Regularization parameter for the sparse component.
    Returns:
        Updated low-rank and sparse components, and dual variable.
    """
    for it in range(iter_max):
        U, s, Vt = torch.linalg.svd(X - S + Y / rho, full_matrices=False)
        _s = soft_threshold(s, alpha/rho)
        L = U @ torch.diag(_s) @ Vt
        S = soft_threshold(X - L + Y/rho, beta/rho)
        Y = Y + rho * (X - L - S)
        if torch.linalg.norm(X - L - S, 'fro') < tol:
            break
    return L, S, Y

def train(model,
          loss_fn,
          scheduler,
          num_epochs,
          train_loader,
          optimizer,
          alpha,
          beta,
          rho,
          iter_max=100,
          tol=1e-3,
          name_file='model'):

    path_model = os.path.join(root, 'data', 'LowSpa', name_file + '.pth')
    path_matrix = os.path.join(root, 'data', 'LowSpa', name_file + '_matrix')

    model.train()
    train_losses = []

    X1 = model.fc1.weight
    X2 = model.fc2.weight
    
    U1, s1, Vt1 = torch.linalg.svd(X1, full_matrices=False)
    U2, s2, Vt2 = torch.linalg.svd(X2, full_matrices=False)
    nr_singular_values1 = int(len(s1) * 0.3)
    nr_singular_values2 = int(len(s2) * 0.3)
    L1 = U1[:, :nr_singular_values1] @ torch.diag(s1[:nr_singular_values1]) @ Vt1[:nr_singular_values1, :]
    L2 = U2[:, :nr_singular_values2] @ torch.diag(s2[:nr_singular_values2]) @ Vt2[:nr_singular_values2, :]
    model.fc1.weight.data = L1.clone()
    model.fc2.weight.data = L2.clone()
    # Initialize low-rank, sparse components and dual variables
    # L1 = torch.zeros_like(X1, device=device)
    # L2 = torch.zeros_like(X2, device=device)
    S1 = torch.zeros_like(X1, device=device)
    S2 = torch.zeros_like(X2, device=device)
    Y1 = torch.zeros_like(X1, device=device)
    Y2 = torch.zeros_like(X2, device=device)

    # L1, S1, Y1 = PRCA(X1.detach(), L1, S1, Y1, al, mu, la[0])
    # L2, S2, Y2 = PRCA(X2.detach(), L2, S2, Y2, al, mu, la[1])
    
    for epoch in range(num_epochs):
        epoch_loss = 0.
        for batch_idx, (data, y) in enumerate(train_loader):
            data = data.to(device)
            y = y.to(device)
            # zero grad
            optimizer.zero_grad()
            output = model(data)
            loss1 = get_loss_term(X1, L1, S1, Y1, rho[0])
            loss2 = get_loss_term(X2, L2, S2, Y2, rho[1])
            loss0 = loss_fn(output, y)
            loss = loss0 + loss1 + loss2

            # backward
            loss.backward()
            # update step
            optimizer.step()

            iter_loss = loss0.item()
            epoch_loss += iter_loss

            # S1 = X1.detach() - L1.detach()
            # S2 = X2.detach() - L2.detach()

            L1, S1, Y1 = PRCA(X1.detach(), L1, S1, Y1, alpha[0], beta[0], rho[0], iter_max=iter_max, tol=tol)
            L2, S2, Y2 = PRCA(X2.detach(), L2, S2, Y2, alpha[1], beta[1], rho[1], iter_max=iter_max, tol=tol)

            print(f'E:{epoch}({100. * batch_idx / len(train_loader):.0f}%) | '
                  f'L0: {iter_loss:.6f} | '
                  f'L1: {loss1.item():.6f} | '
                  f'L2: {loss2.item():.6f} | '
                  f'l1: {torch.linalg.norm(X1 - L1 - S1, "fro"):.6f} | '
                  f'l2: {torch.linalg.norm(X2 - L2 - S2, "fro"):.6f} | '
                  f'NZ1: {torch.count_nonzero(S1)}/{S1.numel()}({100.0 * torch.count_nonzero(S1)/S1.numel():.2f}%)| '
                  f'NZ2: {torch.count_nonzero(S2)}/{S2.numel()}({100.0 * torch.count_nonzero(S2)/S2.numel():.2f}%)')
                  
        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        train_losses.append(mean_epoch_loss)
        scheduler.step()
        # save model
        torch.save(model.state_dict(), path_model)
        # save matrices
        data = {
            'L1': L1.cpu().numpy(),
            'S1': S1.cpu().numpy(),
            'L2': L2.cpu().numpy(),
            'S2': S2.cpu().numpy()
        }
        with open(path_matrix, 'wb') as file:
            pickle.dump(data, file)

    return train_losses

def main(batch_size: int,
         num_epochs: int,
         lr: float,
         alpha: list,
         beta: list,
         rho: list,
         iter_max=100,
         tol=1e-3,
         step_size=5,
         gamma=0.7):
    # get the dataloader
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    # import models
    cnn = CNN().to(device)
    # define the optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    name_file = str(batch_size) + '_' + str(num_epochs) + '_' + str(lr) + \
          '_' + str(alpha) + '_' + str(beta) + '_' + str(rho) + '_' + str(iter_max) + \
          '_' + str(tol) + '_' + str(step_size) + '_' + str(gamma)

    train_losses= train(cnn,
                        loss_fn,
                        scheduler,
                        num_epochs,
                        train_loader,
                        optimizer,
                        alpha=alpha,
                        beta=beta,
                        rho=rho,
                        iter_max=iter_max,
                        tol=tol,
                        name_file=name_file)
    
    plt.yscale('log')
    plt.plot(train_losses)
    plt.show()


if __name__ == '__main__':
    # main(batch_size=512,
    #      num_epochs=100,
    #      lr=0.008,
    #      alpha=[0.12, 0.10],
    #      beta=[0.008, 0.020],
    #      rho=[0.01, 0.01],
    #      iter_max=1,
    #      tol=1e-3,
    #      step_size=5,
    #      gamma=0.7)
    
    main(batch_size=512,
         num_epochs=100,
         lr=0.008,
         alpha=[0.0001, 0.00050],
         beta=[0.000008, 0.00010],
         rho=[0.0001, 0.001],
         iter_max=1,
         tol=1e-3,
         step_size=5,
         gamma=0.7)
    
