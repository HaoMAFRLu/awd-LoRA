"""This script tests how to distill a pretrained model.
"""
import detectors
import sys, os
import torch
import timm
import matplotlib.pyplot as plt

from awave.utils.misc import get_wavefun
from awave.transform2d import DWT2d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.resnet import ResNet18
from datasets.dataloader import get_cifar10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # load the pretrained model
    path_file = '/home/hao/Desktop/MPI/awd-LoRA/models/pretrained/pytorch_model.bin'
    model = timm.create_model("resnet18_cifar10", pretrained=False, num_classes=10)
    state_dict = torch.load(path_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    wt = DWT2d(wave='db5', J=4, device=device)
    phi_orig, psi_orig, x_orig = get_wavefun(wt)

    _, test_loader = get_cifar10()
    
    wt.fit(train_loader=test_loader, 
           pretrained_model=model,
           lr=1e-1, 
           num_epochs=2,
           lamL1attr=5) # control how much to regularize the model's attributions


    phi, psi, x = get_wavefun(wt)
    plt.plot(x, psi)
    plt.plot(x_orig, psi_orig)
    plt.show()

if __name__ == '__main__':
    main()