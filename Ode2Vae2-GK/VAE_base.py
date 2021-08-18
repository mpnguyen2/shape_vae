import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Base Class for all VAEs
class VAE(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    # Encode x
    def encode(self, x):
        return
    
    # Decode latent z
    def decode(self, z):
        return

    # Forward
    def forward(self, x):
        return

    # Reparameterization trick to sample from mu and logvar
    def reparameterize(self, mu, logvar, L=torch.FloatTensor([])):
        
        if mu.size()[0] == 0:
            return torch.FloatTensor([])
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if L.size()[0] == 0:
            return mu + eps * std
        else:
            return mu + torch.matmul(L, eps.reshape(-1, mu.size()[-1], 1 )).reshape(mu.size()) + std * eps