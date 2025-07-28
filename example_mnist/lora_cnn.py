import os
import sys
import numpy as np
import torch
import random
import pickle
import argparse
import torch.nn as nn
import torch.nn.functional as F

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from awave.losses import get_loss_f
from awave.utils.train import Trainer
from awave.utils.evaluate import Validator
from awave.transform2d import DWT2d
from awave.utils.warmstart import warm_start

from models.cnn import CNN
from datasets.dataloader import get_mnist
from utils.general import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = get_parent_path(lvl=1)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def get_pretrained():
    model = CNN().to(device)
    path_file = os.path.join(root, 'models', 'pretrained', 'CNN.pth')
    state_dict = torch.load(path_file, map_location=device)
    model.load_state_dict(state_dict)

    model = model.eval()
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False

    return model

def validate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%')
    return accuracy

def svd_fc(layer, rank):
    """Apply SVD to a fully connected layer and return the compressed version."""
    weight = layer.weight.data
    u, s, v = torch.svd(weight)
    compressed_weight = torch.mm(u[:, :rank], torch.diag(s[:rank]))
    compressed_layer = nn.Linear(compressed_weight.size(1), layer.out_features)
    compressed_layer.weight.data = compressed_weight
    compressed_layer.bias.data = layer.bias.data
    return compressed_layer

def main(batch_size,
         num_epochs,
         lr):
    # get the dataloader
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    # get the pretrained model
    model = get_pretrained()

    linear_layers = [model.fc1, model.fc2]
    compressed_fc = svd_fc(linear_layers[1], rank=10)

    accuracy = validate(model, train_loader)
    accuracy = validate(model, test_loader)
   

if __name__ == '__main__':
    main(batch_size=128,
         num_epochs=50,
         lr=0.001)


