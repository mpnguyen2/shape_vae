"""
Testing files.
"""
import torch
from torch import nn
from train2 import train
from net_model import LatentRL, VAE
import time, logging
'''
Do some testing here
'''

#### TESTING TRAIN ####
train(num_iter_pretrain_vae = 0, \
      num_eps_with_VAE=500, num_eps_without_VAE=200, initialize_interval=10, \
          T = 180, discounted_rate = 0.05, log_interval=20, learning_rate = 3e-2)
