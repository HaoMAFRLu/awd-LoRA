import os
import sys
import numpy as np
import torch
import random
import pickle

from awave.losses import get_loss_f
from awave.utils.train import Trainer
from awave.utils.evaluate import Validator
from awave.transform2d import DWT2d
from awave.utils.warmstart import warm_start

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Current working dir:", os.getcwd())
print("Script dir:", os.path.dirname(__file__))
print("sys.path:", sys.path)
print("CNN module path:", os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn.py')))

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

def get_wave_transform(is_warmstart,
                       wave,
                       mode,
                       J,
                       init_factor,
                       noise_factor,
                       const_factor):
    if is_warmstart is False:
        wt = DWT2d(wave=wave, mode=mode, J=J,
                    init_factor=init_factor,
                    noise_factor=noise_factor,
                    const_factor=const_factor,
                    device=device)
    else:
        pass
        # wt = warm_start(p, out_dir).to(device)
    wt.train()
    return wt

def main(batch_size,
         is_warmstart,
         num_epochs,
         wave,
         mode,
         J,
         init_factor,
         noise_factor,
         const_factor,
         lr,
         target,
         attr_methods,
         lamlSum: float = 1.,
         lamhSum: float = 1.,
         lamL2norm: float = 1.,
         lamCMF: float = 1.,
         lamConv: float = 1.,
         lamL1wave: float = 1.,
         lamL1attr: float = 1.):
    folder_name = get_folder_name()
    path_folder = os.path.join(root, 'data', 'awd_training', folder_name)
    mkdir(path_folder)

    # get the dataloader
    train_loader, test_loader = get_mnist(batch_size=batch_size)
    # get the pretrained model
    model = get_pretrained()
    wt = get_wave_transform(is_warmstart,
                            wave,
                            mode,
                            J,
                            init_factor,
                            noise_factor,
                            const_factor,)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        wt = torch.nn.DataParallel(wt)
    

    # train
    params = list(wt.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_f = get_loss_f(lamlSum=lamlSum, 
                        lamhSum=lamhSum, 
                        lamL2norm=lamL2norm, 
                        lamCMF=lamCMF, 
                        lamConv=lamConv,
                        lamL1wave=lamL1wave, 
                        lamL1attr=lamL1attr)
    trainer = Trainer(model, 
                      wt, 
                      optimizer, 
                      loss_f, 
                      target=target,
                      use_residuals=True, 
                      attr_methods=attr_methods, 
                      device=device, 
                      n_print=1)

    # run
    trainer(train_loader, epochs=num_epochs)

    # calculate losses
    print('calculating losses and metric...')
    validator = Validator(model, test_loader)
    rec_loss, lsum_loss, hsum_loss, L2norm_loss, CMF_loss, conv_loss, L1wave_loss, L1saliency_loss, L1inputxgrad_loss = validator(wt, target=target)
    
    data = {
        'batch_size': batch_size,
        'is_warmstart': is_warmstart,
        'num_epochs': num_epochs,
        'wave': wave,
        'mode': mode,
        'J': J,
        'init_factor': init_factor,
        'noise_factor': noise_factor,
        'const_factor': const_factor,
        'lr': lr,
        'target': target,
        'attr_methods': attr_methods,
        'lamlSum': lamlSum,
        'lamhSum': lamhSum,
        'lamL2norm': lamL2norm,
        'lamCMF': lamCMF,
        'lamConv': lamConv,
        'lamL1wave': lamL1wave,
        'lamL1attr': lamL1attr,
        'train_losses': trainer.train_losses,
        'rec_loss': rec_loss,
        'lsum_loss': lsum_loss,
        'hsum_loss': hsum_loss,
        'L2norm_loss': L2norm_loss,
        'CMF_loss': CMF_loss,
        'conv_loss': conv_loss,
        'L1wave_loss': L1wave_loss,
        'L1saliency_loss': L1saliency_loss,
        'L1inputxgrad_loss': L1inputxgrad_loss,
        'net': wt
    }

    path_file = os.path.join(path_folder, 'params')
    with open(path_file, 'wb') as file:
        pickle.dump(data, file)

    path_file = os.path.join(path_folder, 'model.pth')
    torch.save(wt.state_dict(), path_file)


if __name__ == '__main__':
    main(batch_size=128,
         is_warmstart=False,
         num_epochs=100,
         wave='db5',
         mode='zero',
         J=4,
         init_factor=1,
         noise_factor=0,
         const_factor=0,
         lr=0.1,
         target=6,
         attr_methods='Saliency',
         lamlSum=1.0,
         lamhSum=1.0,
         lamL2norm=1.0,
         lamCMF=1.0,
         lamConv=1.0,
         lamL1wave=1.0,
         lamL1attr=5.0)


