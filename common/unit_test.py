import numpy as np
import torch
import scipy

import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

#import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
import utils as U

initialize_all, initialize_subgrid = False, True
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