""" 
This file consists of description of all network as well as forward definition
"""

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

print(torch.__version__)

class VAE(nn.Module):
    def __init__(self, latent_dims=8, input_channels=1):
        super(VAE, self).__init__()
        # To be filled: Just a standard VAE: Convolutional net and upsampling
        # x dim: (32, 32) (is a subgrid of big image X: (512, 512))
        # z1 dim: 8 is the latent state in this subgrid VAE.
        # z1 is just part of z. x maps to z1, while X maps to z (of dim 16*16*8)
        
        self.latent_dims = latent_dims

        # Initialize Encoder Conv Layer
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2), # 15 * 15 * 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # 7 * 7 * 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2), # 3 * 3 * 128
            nn.ReLU(),
            nn.Flatten() # 1152 
        )       
        # Fully connected to predict mean and variance
        self.mu_fc = nn.Linear (3 * 3 * 128, latent_dims)
        self.logvar_fc = nn.Linear (3 * 3 * 128, latent_dims)
        # Fully connected to send convert latent space back up to the encoder output
        self.fc_decoder = nn.Linear (latent_dims, 3 * 3 * 128)

        # Initialize Decoder Conv Layers
        self.decoder = nn.Sequential(
            nn.Unflatten(1, torch.Size([128, 3, 3])),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2),
            nn.Sigmoid(),
        )


        
        # Store set of x and z1 for traing 
        self.x_buf = []
        self.mu_buf = []
        self.logvar_buf = []
        
    def encode(self, x):
        z_aug = self.encoder(x)
        mu = self.mu_fc(z_aug)
        logvar = self.logvar_fc(z_aug)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        # For now, just do Gaussian output. 
        # One can later consider normalizing flow like IAF
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z1):
        new_z = self.fc_decoder(z1)
        print(new_z.shape)
        x  = self.decoder(new_z)
        return x
    
    # forward may not be needed ????
    def forward(self, x):
        mu, logvar = self.encode(x)
        z1 = self.reparameterize(mu, logvar)
        return self.decode(z1), mu, logvar
        

class LatentRL(nn.Module):
    def __init__(self):
        super(LatentRL, self).__init__()
        # To be filled
        self.saved_actions = []
        self.rewards = []
        self.net_rewards = []   # Reward returned by the artificial net

    def actor(self):
        return

    def critic(self):
        return
    
    def reward_net(self):
        return
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



if __name__ == "__main__":
    print("Testing model")
    vae = VAE()
    summary(vae, (1, 32, 32))