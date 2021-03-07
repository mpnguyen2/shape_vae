""" 
This file consists of description of all network as well as forward definition
"""

import torch
from torch import nn, optim
from torch.nn import functional as F

print(torch.__version__)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # To be filled: Just a standard VAE: Convolutional net and upsampling
        # x dim: (32, 32) (is a subgrid of big image X: (512, 512))
        # z1 dim: 8 is the latent state in this subgrid VAE.
        # z1 is just part of z. x maps to z1, while X maps to z (of dim 16*16*8)
        
        # Store set of x and z1 for traing 
        self.x_buf = []
        self.mu_buf = []
        self.logvar_buf = []
        
    def encode(self, x):
        # To be filled
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        # For now, just do Gaussian output. 
        # One can later consider normalizing flow like IAF
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z1):
        # To be filled
        return x
    
    # forward may not be needed ????
    def forward(self, x):
        mu, logvar = self.encode(x)
        z1 = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        

class LatentRL(nn.Module):
    def __init__(self):
        super(LatentRL, self).__init__()
        # To be filled
        self.saved_actions = []
        self.rewards = []
        self.net_rewards = []   # Reward returned by the artificial net
        
    def forward(self, z):
        # To be filled
        """
        Consist of 3 nets that take z as input Return the followings:
        1. action (policy): which include a pairs of (p, v)
        p indicate the position of the subgrid action act on
        It is represented by a 4 by 4 matrix where each column 
        represent categorial probability distribution on 4 items 
        each item represent a quadrant
        v with dim 8 indicates how much action is exerted on 
        action is then a 4*4 + 4 = 20 dim vector
        2. value function:       
        3. reward function:
        All these nets share parts of the net starting from z
        so we group them here inside LatentRL
        """
        return action, value, net_reward