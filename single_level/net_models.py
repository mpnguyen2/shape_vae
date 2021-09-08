import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from common_nets import Mlp
from torch.distributions import multivariate_normal
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class ACSimple(nn.Module):
    def __init__(self):
        super(ACSimple, self).__init__()
        self.log_prob = []
        self.values = []
        self.rewards = []
        # Reward returned by the artificial net
        #self.net_rewards = []  

        # Predicts the action vector 
        # Input: 80, Output: 80
        latent_dim = 80
        self.latent_dim = latent_dim
        self.actor = nn.Mlp()
        self.critic = nn.Mlp()
    
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
