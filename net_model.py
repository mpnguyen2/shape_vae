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
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), # 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 4 x 4 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten() # 2048
        )       
        # Fully connected to predict mean and variance
        self.mu_fc = nn.Linear (2048, latent_dims)
        self.logvar_fc = nn.Linear (2048, latent_dims)
        # Fully connected to send convert latent space back up to the encoder output
        self.fc_decoder = nn.Linear (latent_dims, 2048)

        #Initialize Decoder Conv Layers
        self.decoder = nn.Sequential(
            nn.Unflatten(1, torch.Size([128, 4, 4])), # 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 64 x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1), # 1 x 32 x 32
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
        temp = self.fc_decoder(z1)
        x  = self.decoder(temp)
        return x
    
    # forward may not be needed ????
    def forward(self, x):
        mu, logvar = self.encode(x)
        z1 = self.reparameterize(mu, logvar)
        recon_x = self.decode(z1)
        
        return 0, mu, logvar
        

class LatentRL(nn.Module):
    def __init__(self):
        super(LatentRL, self).__init__()
        # To be filled
        self.saved_actions = []
        self.rewards = []
        self.net_rewards = []   # Reward returned by the artificial net

        # Initializes shared part of the network that processes the z-matrix of subgrids
        # Input: 8 x 16 x 16, Output: 32 x 4 x 4 
        self.subgrids_net = nn.Sequential(                               
           nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16 x 8 x 8
           nn.BatchNorm2d(16),
           nn.ReLU(),
           nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 x 4 x 4
           nn.BatchNorm2d(32),
           nn.ReLU(),
        )

        # Actor network components

        self.actor_shared = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 2 x 2
            nn.BatchNorm2d(64),
            nn.ReLU(),              
            nn.Flatten(), # 256
            nn.Linear(256, 128) # 128
            nn.ReLU()
        )

        # Predicts the p-matrix containin quadrant position
        self.p_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 16), 
            nn.Softmax(),
        )

        # Predcits the action vector v
        self.v_net = nn.Sequential(
            nn.Linear
        )
        
        # Critic network which predicts value
        # Input: 8 x 16 x 16, Output: 1 
        self.critic = nn.Sequential (
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 2 x 2
            nn.Flatten(), # 256
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 1)
        )

    def actor(self, z_proc):
        return
    
    def critic(self, z_proc):
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

        # Shared network that processes z
        z_processed = self.subgrids_net(z)
        
        # Get the action 
        #p, v = self.action(z)
        # Predict the value
        #value = self.critic(z)
        # Predict the reward
        #reward_pred = self.reward(z, p, v)
        return z_processed
        #return action, value, net_reward



if __name__ == "__main__":
    print("Testing model")
    vae = VAE()
    agent = LatentRL()
    summary(vae, (1, 32, 32))
    #summary(agent, (8, 16, 16))