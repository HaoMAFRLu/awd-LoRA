import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lowspa_ddp.utils import get_parent_path

root = get_parent_path(lvl=1)

def get_batch(split, 
              batch_size=64, 
              block_size=256, 
              data_dir='data/openwebtext'):

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y

def get_loader(split='train',
               batch_size=64,
               block_size=256,
               data_dir='data/openwebtext'):
    """Generator for training data batches.
    """
    while True:
        yield get_batch(split, 
                        batch_size=batch_size, 
                        block_size=block_size,
                        data_dir=data_dir)

def get_gpt_dataloader(batch_size=128, 
                       block_size=1024):
    """Load GPT train/test dataloaders."""
    dataset = 'openwebtext'
    data_dir = os.path.join(root, 'datasets', 'data', dataset)
    train_loader = get_loader('train', batch_size=batch_size, 
                              block_size=block_size, 
                              data_dir=data_dir)
    test_loader = get_loader('val', batch_size=batch_size, 
                             block_size=block_size, 
                             data_dir=data_dir)
    return train_loader, test_loader
