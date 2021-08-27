""" 
This file consists of description of all network as well as forward definition
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import multivariate_normal
#from torchsummary import summary
#from torch.utils.tensorboard import SummaryWriter

#print(torch.__version__)
# GPU to use
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

# Deterministic AE
class AE(nn.Module):
    def __init__(self, latent_dim=16, input_channels=1):
        super(AE, self).__init__()
        # x dim: (32, 32)
        # latent z dim: latent_dim
        
        # Store set of x and x_recon for traing 
        #self.x_buf = []
        #self.x_recon_buf = []
        # Initialize Encoder Conv Layer
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 4, kernel_size=3, stride=2, padding=1), # 4 x 16 x 16
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1), # 8 x 8 x 8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16 x 4 x 4 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 x 2 x 2 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten() # 128
        ) 
        
        # Fully connected to send convert latent space back up to the encoder output
        self.fc_mu = nn.Sequential(
            nn.Linear(128, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(128, latent_dim)
        )
        
        # FC decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
        )
        
        #Initialize Decoder Conv Layers
        self.decoder = nn.Sequential(
            nn.Unflatten(1, torch.Size([32, 2, 2])), # 32 x 2 x 2
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # 16 x 4 x 4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1), # 8 x 8 x 8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1), # 4 x 16 x 16
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ConvTranspose2d(4, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # 1 x 32 x 32 
            nn.Sigmoid(),
        )
        
    def encode(self, x):
        z_aug = self.encoder(x)
        return self.fc_mu(z_aug), self.fc_logvar(z_aug)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
        
    def decode(self, z):
        return self.decoder(self.fc_decoder(z))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z

# Approximate reward function as a fct of latent variable     
class Reward(nn.Module):
    def __init__(self):
        super(Reward, self).__init__()
        latent_dim = 80
        self.reward = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.ReLU()
        )
    
    def forward(self, z):
        return self.reward(z)
        
class LatentRL(nn.Module):
    def __init__(self):
        super(LatentRL, self).__init__()
        self.log_prob = []
        self.values = []
        self.rewards = []
        # Reward returned by the artificial net
        #self.net_rewards = []  

        # Predicts the action vector 
        # Input: 80, Output: 80
        latent_dim = 80
        self.latent_dim = latent_dim
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 2),
            nn.LeakyReLU(),
            nn.Linear(2, 1),
            nn.ReLU()
        )
    
    def forward(self, z):
        # To be filled
        """
        Take z as input and return the followings:
        1. action (policy) v: dim 64
        2. value function:       
        3. reward function: (TBA)
        """            
        
        action_mean = self.actor(z).squeeze(0)
        #action_logvar = self.actor_logvar(z).squeeze(0)
        action_dist = multivariate_normal.MultivariateNormal(loc=action_mean,\
            covariance_matrix=0.04*torch.eye(self.latent_dim).to(device))
        
        act = action_dist.sample()
        self.log_prob.append(action_dist.log_prob(act))
        value = self.critic(z)
        self.values.append(value)
        return act.unsqueeze(0)
