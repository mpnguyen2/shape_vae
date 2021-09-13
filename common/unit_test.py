import numpy as np
import torch
import scipy

import matplotlib.pyplot as plt
import time

from common.common_nets import VAE
from common.utils import get_samples

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

device = "cpu"

#import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
import utils as U

initialize_all, initialize_subgrid = False, False
if initialize_all:
    area, peri, total_area, total_peri, X = U.initialize_single(
                grid_dim=8, subgrid_dim=16, mode='mountain', 
                num_milestone=11, spacing=6, start_horizontal=10,
                xg=None, yg=None)
    plt.imshow(X, cmap='gray')

if initialize_subgrid:
    x = U.get_sample(subgrid_dim=16, num_milestone=5, spacing=3)
    plt.imshow(x, cmap='gray')
    x_dat = U.get_batch_sample(batch_size=10, subgrid_dim=16, num_milestone=5, spacing=3)
    print(x_dat.shape, type(x_dat))

test_vae=False
if test_vae:
    subgrid_dim = 16; latent_dim = 16; num_samples = 15
    X = get_samples(num_samples=num_samples, subgrid_dim=subgrid_dim, num_milestone=5, spacing=3)
    vae_model = VAE(img_dim=subgrid_dim, latent_dim=latent_dim, channels=[1, 4, 8, 16]).to(device)
    vae_model.load_state_dict(torch.load('vae_model'))
    
    for i in range(num_samples):
        plt.imshow(X[i].detach().numpy(), cmap='gray')
        plt.show()
        X_recon,_,_ = vae_model(X[i].unsqueeze(0))
        plt.imshow(X_recon.detach().numpy().squeeze(), cmap='gray')
        plt.show()
