import torch

print("Torch version:", torch.version)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Number of GPUs:", torch.cuda.device_count())