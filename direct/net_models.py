import gym
import torch
from torch import nn

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.policies import ActorCriticPolicy

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from common.common_nets import Mlp

## PPO CUSTOM NET from stable_baselines3 library ##
class PolicyNet(nn.Module):
    """
    Network that output from the common extractor for actor as well as critic networks for on-policy training
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 128,
        last_layer_dim_vf: int = 16,
    ):
        super(PolicyNet, self).__init__()
        
        # IMPORTANT: Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = Mlp(input_dim=feature_dim, output_dim=last_layer_dim_pi,
            layer_dims=[32, 64]) 
        
        # Value network
        self.value_net = Mlp(input_dim=feature_dim, output_dim=last_layer_dim_vf,
            layer_dims=[64, 16, 8, 4]) 

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

# Shape Policy for PPO training. No common feature extractor used (for actor and critic nets)
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