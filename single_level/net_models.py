import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import multivariate_normal

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

# Import from common. If pip install no need to do this
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from common.common_nets import Mlp, CNNEncoder, CNNDecoder

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

# State {subgrid: 1 x 16 x 16; grid_compressed: 16 x 8 x 8}
# Action {subgrid_change: 1 x 16 x 16; position probabilities p: 64}
class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=16*2*2+256):
        super(Extractor, self).__init__(observation_space, features_dim)
        self.subgrid_encoder = CNNEncoder(channels=[1, 4, 8, 16])
        self.grid_extractor = CNNEncoder(channels=[16, 32, 64])
        
    def forward(self, observations):
        change = self.subgrid_encoder(observations['subgrid'].unsqueeze(1))
        p = self.grid_extractor_mlp(self.grid_extractor(observations['grid_compressed']))
        
        return torch.cat((change, p), dim=1)
        
# Redefine policy net
class ActorNet(nn.Module):
    """
    Actor net for PPO training

    Args:
        feature_dim(int): dimension of the features extracted with the features_extractor
   
    """
    def __init__(self, feature_dim):
        super(ActorNet, self).__init__()
        self.change_dim = 16*2*2
        self.subgrid_decoder = CNNDecoder(channels=[16, 8, 4, 1], img_dim=16)
        self.grid_position_mlp = Mlp(input_dim=feature_dim-self.change_dim, 
                                     output_dim=64, layer_dims=[128, 64, 32, 256])
        self.sigmoid_layer = nn.Sigmoid()
        
    def forward(self, features):    
        return dict({'change': self.subgrid_decoder(features[:self.change_dim]), 
                     'position': self.sigmoid_layer(self.grid_position_mlp(features[self.change_dim:]))})

class PolicyNet(nn.Module):
    """
    Policy (actor, value) net for PPO training. SAC to be added

    Args:
        feature_dim(int): dimension of the features extracted with the features_extractor
   
    """

    def __init__(self, feature_dim):
        super(PolicyNet, self).__init__()

        # Policy network
        self.policy_net = ActorNet(feature_dim)
        
        # Value network
        self.value_net = Mlp(input_dim=feature_dim, output_dim=8, layer_dims=[256, 64, 16, 32])

    def forward(self, features):
        """
        Return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
        """
        return self.policy_net(features), self.value_net(features)


class ShapePPOPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(ShapePPOPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PolicyNet(self.features_dim)


''' Others

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
'''