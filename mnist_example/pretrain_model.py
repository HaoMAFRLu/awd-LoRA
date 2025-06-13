"""This script is used to pretrain a CNN for 
late
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os, sys
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn import CNN
from datasets.dataloader import get_mnist
from utils.general import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = get_parent_path(lvl=1)

def train(model,
          loss_fn,
          num_epochs,
          train_loader,
          optimizer):
    
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.
        for batch_idx, (data, y) in enumerate(train_loader):
            data = data.to(device)
            y = y.to(device)
            # zero grad
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, y)

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

    # save model
    model_path = os.path.join(root, 'models', 'pretrained', 'CNN.pth')
    torch.save(model.state_dict(), model_path)
    return train_losses

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

    print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({100. * accuracy:.2f}%)\n')
    return avg_loss, accuracy

def main(batch_size,
         num_epochs,
         lr):
    # get the dataloader
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    # import models
    cnn = CNN().to(device)
    # define the optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_losses = train(cnn,
                         loss_fn,
                         num_epochs,
                         train_loader,
                         optimizer)
    
    eval_loss, accuracy = evaluate(cnn,
                                   test_loader,
                                   loss_fn)
    
    plt.plot(train_losses)
    plt.show()

if __name__ == '__main__':
    main(batch_size=128,
         num_epochs=100,
         lr=0.001,)