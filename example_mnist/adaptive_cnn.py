import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
import pickle

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from awave.transform2d import DWT2d

from models.cnn import CNN
from datasets.dataloader import get_mnist
from utils.general import *
from models.wave_cnn import Wave_CNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = get_parent_path(lvl=1)

def get_params(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def _get_w_transform(params, path):
    wt = DWT2d(wave=params['wave'], 
               mode=params['mode'], 
               J=params['J'], 
               init_factor=params['init_factor'], 
               noise_factor=params['noise_factor'],
               const_factor=params['const_factor'],
               device=device)
    wt.load_state_dict(torch.load(path))
    return wt

def get_w_transform(path):
    params = get_params(os.path.join(path, 'params'))
    wt = _get_w_transform(params, os.path.join(path, 'model.pth'))
    return wt

def main(batch_size,
         num_epochs,
         path,
         lr):
    # get the dataloader
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    # get pretrained cnn
    model = get_pretrained()
    # get the pretrained wave transform
    wt = get_w_transform(path)
    wcnn = Wave_CNN(wt).to(device)
    optimizer = torch.optim.Adam(wcnn.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.
        for batch_idx, (data, y) in enumerate(train_loader):
            data = data.to(device)
            y = y.to(device)
            # zero grad
            optimizer.zero_grad()
            output = wcnn(data)
            loss = criterion(output, y)

            # backward
            loss.backward()
            # update step
            optimizer.step()

            iter_loss = loss.item()
            epoch_loss += iter_loss

            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), iter_loss), end='')

        mean_epoch_loss = epoch_loss / (batch_idx + 1)
        train_losses.append(mean_epoch_loss)
    
    plt.plot(train_losses)
    plt.show()

    # check prediction
    m = len(test_loader.dataset)
    batch_size = test_loader.batch_size

    y_pred_cnn = np.zeros(m)
    y_pred_wcnn = np.zeros(m)
    y_true = np.zeros(m)
    with torch.no_grad():
        for batch_idx, (data, y) in tqdm(enumerate(test_loader, 0), total=int(np.ceil(m / batch_size))):
            data = data.to(device)
            # cnn prediction
            outputs_cnn = model(data)
            _, y_pred = torch.max(outputs_cnn.data, 1)
            y_pred_cnn[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()

            # ffn prediction
            outputs_ffn = wcnn(data)
            _, y_pred = torch.max(outputs_ffn.data, 1)
            y_pred_wcnn[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y_pred.cpu().numpy()

            # labels
            y_true[batch_idx * batch_size:(batch_idx + 1) * batch_size] = y.numpy()

    print("CNN accuracy {:.5f}% wCNN accuracy {:.5f}%".format((y_true == y_pred_cnn).sum() / m * 100,
                                                            (y_true == y_pred_wcnn).sum() / m * 100))
if __name__ == '__main__':
    folder = 'wave_0.001_attr_1.0'
    path = os.path.join(root, 'data', 'awd_training', folder)
    main(batch_size=128,
         num_epochs=100,
         path=path,
         lr=0.0001)
