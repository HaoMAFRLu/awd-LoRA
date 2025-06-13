"""This script tests how to load local model.
"""
import detectors
import sys, os
import torch
import timm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.resnet import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    path_file = '/home/hao/Desktop/MPI/awd-LoRA/models/pretrained/pytorch_model.bin'
    model = timm.create_model("resnet18_cifar10", pretrained=False, num_classes=10)
    state_dict = torch.load(path_file, map_location=device)
    model.load_state_dict(state_dict)

    print('here')


    features = {}
    def hook_fn(name):
        def hook(m, i, o):
            features[name] = o.detach()
        return hook

    model.layer2[1].register_forward_hook(hook_fn("layer2_block1"))
if __name__ == '__main__':
    main()