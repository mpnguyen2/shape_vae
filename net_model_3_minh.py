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
    def __init__(self, latent_dims=8, input_channels=1):
        super(VAE, self).__init__()
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
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 4 x 4 
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.Flatten() # 2048
        )       
        # Fully connected to predict mean and variance
        self.mu_fc = nn.Linear(2048, latent_dims)
        self.logvar_fc = nn.Linear(2048, latent_dims)
        # Fully connected to send convert latent space back up to the encoder output
        self.fc_decoder = nn.Linear(latent_dims, 2048)
        
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
        self.x_recon_buf = []
        self.mu_buf = []
        self.logvar_buf = []
        
    def encode(self, x):
        z_aug = self.encoder(x)
        mu = self.mu_fc(z_aug)
        logvar = self.logvar_fc(z_aug)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        # For now, just do Gaussian output. Consider normalizing flow such as IAF later
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z1):
        temp = self.fc_decoder(z1)
        x  = self.decoder(temp)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z1 = self.reparameterize(mu, logvar)
        return self.decode(z1), mu, logvar
        

class LatentRL(nn.Module):
    def __init__(self):
        super(LatentRL, self).__init__()
        self.saved_actions = []
        self.rewards = []
        # Reward returned by the artificial net
        self.net_rewards = []  
        
        # Input: 8 x 16 x 16, Output: 256
        self.subgridl1 = nn.Sequential(                               
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16 x 8 x 8
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 x 4 x 4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64 x 2 x 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten() # 256
        )
        
        # Input: 256, Output: 4 (probability: sum to 1)
        self.quadl1 = nn.Sequential(
            nn.Linear(256, 128), # 128
            nn.ReLU(),
            nn.Linear(128, 64), # 64
            nn.ReLU(),
            nn.Linear(64, 16), # 16
            nn.ReLU(),
            nn.Linear(16, 4), # 4
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        
        # Input: 8 x 8 x 8, Output: 64
        self.subgridl2 = nn.Sequential(                               
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16 x 4 x 4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 x 2 x 2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten() # 128
        )
        
        # Input: 128, Output: 4 (probability: sum to 1)
        self.quadl2 = nn.Sequential(
            nn.Linear(128, 64), # 64
            nn.ReLU(),
            nn.Linear(64, 16), # 16
            nn.ReLU(),
            nn.Linear(16, 4), # 4
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        
        # Input: 8 x 4 x 4, Output: 64
        self.subgridl3 = nn.Sequential(                               
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16 x 2 x 2
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten() # 64
        )
        
        # Input: 128, Output: 4 (probability: sum to 1)
        self.quadl3 = nn.Sequential(
            nn.Linear(64, 32), # 32
            nn.ReLU(),
            nn.Linear(32, 16), # 16
            nn.ReLU(),
            nn.Linear(16, 4), # 4
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        
        # Input: 8 x 2 x 2, Output: 4 (probability: sum to 1)
        self.quadl4 = nn.Sequential(
            nn.Flatten(), # 32
            nn.Linear(32, 16), # 16
            nn.ReLU(),
            nn.Linear(16, 8), # 8
            nn.ReLU(),
            nn.Linear(8, 4), # 4
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

        # Predicts the action vector 
        # Input: 128 Output: 8
        self.v_net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        )
        
        # Critic network which predicts value
        # Input: 256, Output: 1 
        self.critic = nn.Sequential (
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.ReLU()
        )

        # Reward network
        # Process the processed z matrix for the reward network
        self.reward_z_net = nn.Sequential (
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
        
    def forward(self, z):
        # To be filled
        """
        Take z as input and return the followings:
        1. action (policy): which include a pairs of (p, v)
        p indicate the position of the subgrid action act on
        Each part of p represents categorial probability distribution on 4 items, 
        where each item represent a quadrant
        v with dim 8 indicates how much action is exerted on 
        action is then a 4*4 + 8 = 24 dim vector
        2. value function:       
        3. reward function:
        All these nets share parts of the net starting from z
        so we group them here inside LatentRL
        """            
        
        ## Forward p ##
        # Input z: 8 x 16 x 16. Output: prob distribution for level 1 quadrants
        z_common = self.subgridl1(z)
        p1 = self.quadl1(z_common)
        # Input p1 and z, output z_l1: data of the chosen quadrant level 1
        z_l1 = torch.matmul(p1, z.reshape(-1, 2, 8, 2, 8).swapaxes(2, 3).reshape(-1, 4, 64)\
                    .swapaxes(0, 1).reshape(4, -1)).reshape(-1, 8, 8, 8)
        # Input z_l1: 8 x 8 x 8. Output: prob distribution for level 2 quadrants
        p2 = self.quadl2(self.subgridl2(z_l1))
        # Input p1 and z, output z_l2: data of the chosen quadrant level 2
        z_l2 = torch.matmul(p2, z_l1.reshape(-1, 2, 4, 2, 4).swapaxes(2, 3).reshape(-1, 4, 16)\
                    .swapaxes(0, 1).reshape(4, -1)).reshape(-1, 8, 4, 4)
        # Input z_l2: 8 x 4 x 4. Output: prob distribution for level 3 quadrants
        p3 = self.quadl3(self.subgridl3(z_l2))
        # Input p1 and z, output z_l3: data of the chosen quadrant level 3
        z_l3 = torch.matmul(p3, z_l2.reshape(-1, 2, 2, 2, 2).swapaxes(2, 3).reshape(-1, 4, 4)\
                    .swapaxes(0, 1).reshape(4, -1)).reshape(-1, 8, 2, 2)
        p4 = self.quadl4(z_l3)
        
        ## Forward v and concat action (p, v) ##
        # The chosen subcell to change. v, how much to change, depends on z1 only
        z1 = torch.matmul(p4, z_l3.reshape(-1, 2, 1, 2, 1).swapaxes(2, 3).reshape(-1, 4, 1)\
                    .swapaxes(0, 1).reshape(4, -1)).reshape(-1, 8)
        v = self.v_net(z1)
        # Concat p1, p2, p3, p4 into p and then concat p and v
        p = torch.stack([p1, p2, p3, p4]).swapaxes(0, 1).reshape(-1, 16)
        action = torch.cat((p, v), 1)
        
        ## Forward value and artificial reward returned from reward nets ##    
        # Predict the value
        value = self.critic(z_common)
        # Predict the reward
        #print(action.shape)
        net_reward = self.reward(z_common, action)
        
        '''
        # DEBUGGING
        print("z_l1:", z_l1.shape)
        print("p:", p.shape)
        print("v:", v.shape)
        print("Action:", action.shape)
        print("Value:", value.shape)
        print("Reward:", net_reward.shape)
        '''
        return action, z1, value, net_reward


'''
if __name__ == "__main__":
    print("Testing model")
    vae = VAE()
    agent = LatentRL()
    #summary(vae, (1, 32, 32))
    #summary(agent, (8, 16, 16))
'''