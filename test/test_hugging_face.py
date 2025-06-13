"""This script tests how to load model from Hugging Face.
"""
import detectors
import timm
import torch
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.dataloader import get_cifar10
from utils.metrics import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("resnet18_cifar10", pretrained=True).to(device)

_, test_loader = get_cifar10()
acc = evaluate(model, test_loader, device)
print(f"🎯 Test Accuracy: {acc:.4f}")