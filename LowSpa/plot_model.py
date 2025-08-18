"""Generate the plots for the tech meeting.
"""
import matplotlib.pyplot as plt
import os, sys
import torch
from matplotlib.colors import LogNorm 
from scipy.sparse import coo_matrix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn import CNN
from dataloaders.dataloader import get_mnist
from salad.utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
root = get_parent_path(lvl=1)

def plot_kernels(model: CNN):
    """
    Plot the kernels of the CNN model.
    Args:
        model: The CNN model to plot.
    """
    kernels = model.conv2.weight.data.squeeze().cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(kernels[10, i, :, :], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def _plot_rank(X1, X2, L):
    """
    Plot the rank of the CNN model and the baseline model.
    Args:
        baseline: The baseline CNN model to compare against.
        model: The CNN model to plot.
    """
    _, s1, _ = np.linalg.svd(X1)
    _, s2, _ = np.linalg.svd(X2)
    _, s3, _ = np.linalg.svd(L)
    idx = list(range(len(s1)))
    # bar plot
    # plt.figure(figsize=(8, max(3, len(s1)*0.35)))
    plt.figure(figsize=(8, 12))
    set_axes_format(plt.gca(), r'Rank', r'Value')
    plt.barh(idx, s1, alpha=0.3, edgecolor="k", label='Baseline')
    plt.barh(idx, s2, alpha=0.3, edgecolor="k", label='X')
    plt.barh(idx, s3, alpha=0.3, edgecolor="k", label='L')
    plt.legend()
    plt.gca().invert_yaxis()    

    plt.tight_layout()
    plt.show()

def plot_ranks(model: CNN, baseline: CNN, L1, L2):
    """
    Plot the ranks of the CNN model and the baseline model.
    Args:
        model: The CNN model to plot.
        baseline: The baseline CNN model to compare against.
    """
    layers = ['fc1.weight.data', 'fc2.weight.data']
    _plot_rank(baseline.fc1.weight.data, model.fc1.weight.data, L1)
    _plot_rank(baseline.fc2.weight.data, model.fc2.weight.data, L2)

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
    L1 = torch.tensor(data['L1']).to('cpu')
    S1 = torch.tensor(data['S1']).to('cpu')
    L2 = torch.tensor(data['L2']).to('cpu')       
    S2 = torch.tensor(data['S2']).to('cpu')
    return L1, L2, S1, S2

def plot_sparsity(S1, S2): 
    """
    Plot the sparsity of the CNN model.
    Args:
        S1: Sparse matrix for the first layer.
        S2: Sparse matrix for the second layer.
    """
    A = coo_matrix(S2.cpu().numpy())
    plt.figure(figsize=(6,6))
    set_axes_format(plt.gca(), r'Column', r'Row')
    sc = plt.scatter(A.col, A.row, c=abs(A.data), s=1, cmap='viridis',
                    norm=LogNorm(vmin=max(abs(A.data).min(), 1e-12), vmax=abs(A.data).max()))
    plt.gca().invert_yaxis()  # 行 0 在上方，更像矩阵视图
    plt.colorbar(sc, label="|value|")
    plt.tight_layout(); plt.show()

def main(model_file: str=None):
    """
    Main function to generate the plots for the tech meeting.
    Args:
        baseline_model: Name of the baseline model.
        salad_model: Name of the Salad model.
    """
    # baseline model
    baseline = CNN().to('cpu')
    model_path = os.path.join(root, 'models', 'pretrained', 'CNN.pth')
    baseline.load_state_dict(torch.load(model_path, map_location='cpu'))
    # load the model
    model = CNN().to('cpu')
    model_path = os.path.join(root, 'data', 'LowSpa', model_file + '.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # plot_kernels(baseline)
    file_path = os.path.join(root, 'data', 'LowSpa', model_file+'_matrix')    
    L1, L2, S1, S2 = get_matrix(file_path)
    sparcity_S1 = torch.sum(S1 != 0) / S1.numel()
    sparcity_S2 = torch.sum(S2 != 0) / S2.numel()
    print(f"Sparcity of S1: {1 - sparcity_S1:.4f}, Sparcity of S2: {1 - sparcity_S2:.4f}")
    # plot_ranks(model, baseline, L1, L2)
    # plot_sparsity(S1, S2)



if __name__ == "__main__":
    model_file = '512_100_0.008_[0.0001, 0.0005]_[8e-06, 0.0001]_[0.0001, 0.001]_1_0.001_5_0.7'
    main(model_file=model_file)