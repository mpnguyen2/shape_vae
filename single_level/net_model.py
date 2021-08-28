""" 
This file consists of description of all network as well as forward definition
"""

import torch
from torch import nn
from torch.nn import functional as F
#from torchsummary import summary
#from torch.utils.tensorboard import SummaryWriter

#print(torch.__version__)

class VAE(nn.Module):
    def __init__(self, latent_dims=64, input_channels=1):
        super(VAE, self).__init__()
        # x dim: (32, 32) (is a subgrid of big image X: (512, 512))
        # z1 dim: 8 is the latent state in this subgrid VAE.
        # z1 is just part of z. x maps to z1, while X maps to z (of dim 16*16*8)
        
        self.latent_dims = latent_dims

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
        # Fully connected to predict mean and variance
        self.mu_fc = nn.Linear(128, latent_dims)
        self.logvar_fc = nn.Linear(128, latent_dims)
        # Fully connected to send convert latent space back up to the encoder output
        self.fc_decoder = nn.Linear(latent_dims, 128)
        
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
        
        # Store set of x and z1 for traing 
        self.x_buf = []
        self.x_recon_buf = []
        self.mu_buf = []
        self.logvar_buf = []
        
    def encode(self, x):
        z_aug = self.encoder(x.unsqueeze(1))
        mu = self.mu_fc(z_aug)
        logvar = self.logvar_fc(z_aug)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        # For now, just do Gaussian output. Consider normalizing flow such as IAF later
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        temp = self.fc_decoder(z)
        x  = self.decoder(temp)
        return x.squeeze(1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
        

class LatentRL(nn.Module):
    def __init__(self):
        super(LatentRL, self).__init__()
        self.saved_actions = []
        self.rewards = []
        # Reward returned by the artificial net
        #self.net_rewards = []  

        # Predicts the action vector 
        # Input: 64, Output: 64
        self.actor_mean = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64)
        )
        self.actor_logvar = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64)
        )
        
        # Critic network which predicts value
        # Input: 64, Output: 1 
        self.critic = nn.Sequential (
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.ReLU()
        )
        '''
        # Reward network
        # Process the processed z matrix for the reward network
        self.reward_z_net = nn.Sequential (
            nn.Linear(80, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        self.reward_action_net = nn.Sequential (
            nn.Linear(24, 24),
            nn.ReLU()
        )
        # Predicts reward
        # Input: 32 + 24, Output: 1
        self.reward_net_main = nn.Sequential(
            nn.Linear(56, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def reward(self, z_processed, action):
        temp = self.reward_z_net(z_processed)
        temp_act = self.reward_action_net(action)
        #print(temp, temp_act)
        return self.reward_net_main(torch.cat((temp, temp_act), 1))
    '''
    
    def forward(self, z):
        # To be filled
        """
        Take z as input and return the followings:
        1. action (policy) v: dim 80
        2. value function:       
        3. reward function: (TBA)
        """            
        
        action_mean = self.actor_mean(z)
        action_logvar = self.actor_logvar(z)
        
        ## Forward value and artificial reward returned from reward nets ##    
        # Predict the value
        value = self.critic(z)
        # Predict the reward
        #print(action.shape)

        '''
        # DEBUGGING
        print("Action:", action.shape)
        print("Value:", value.shape)
        '''
        return action_mean, action_logvar, value


'''
if __name__ == "__main__":
    print("Testing model")
    vae = VAE()
    agent = LatentRL()
    #summary(vae, (1, 32, 32))
    #summary(agent, (8, 16, 16))
'''