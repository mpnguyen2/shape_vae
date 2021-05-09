#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:23:24 2021

@author: minhpc
"""
import gym
import numpy as np
import torch

from stable_baselines3 import DDPG, A2C, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from net_model import AE

from sb3_model import ShapePolicy, ShapePPOPolicy
from isoperi_env import IsoperiEnv
from isoperi_test_env import IsoperiTestEnv


# Fixed global vars
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#ae_model_path = 'models/ae_model1'
#reward_model_path = 'models/reward_model1'

# Load model
ae_model_path = 'models/ae_mnist1'
ae_model = AE().to(device)
ae_model.load_state_dict(torch.load(ae_model_path))
'''
# Create latent env
env = IsoperiEnv(ae_model=ae_model)
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.0001*np.ones(n_actions))
# action_noise=action_noise,
#policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                     net_arch=dict(pi=[32, 64, 64, 64, 128], qf=[32, 16, 64, 16, 64, 32]))
#model = SAC("MlpPolicy", policy_kwargs=policy_kwargs, env=env, verbose=1) #learning_rate=0.003)
model = DDPG(policy=ShapePolicy, action_noise=action_noise, env=env, verbose=1, learning_rate=0.003)
#model.device = device
model.learn(total_timesteps=100000, log_interval=1)
model.save("ddpg_shape_opt1")
'''

'''
# Create test env
env = IsoperiTestEnv(ae_model)

# Loading model
#model = DDPG.load("ddpg_shape_opt1")
#model.set_env(env)

obs = env.reset()
done = False
while done == False:
    #action, _states = model.predict(obs)
    action = np.random.rand(16)
    obs, rewards, done, info = env.step(action)
    #env.render()
'''

# TEST OTHER RL ALGORITHMS

# Custom actor architecture with two layers of 64 units each
# Custom critic architecture with two layers of 400 and 300 units
#policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                     net_arch=dict(pi=[32, 16, 16, 8, 8, 8, 16, 16, 32], qf=[32, 16, 16, 8, 8, 8, 4, 4, 2]))

#policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                     net_arch=[64, 32, 16, 16, 16, 32, 64])
# Create latent env
'''
env = IsoperiEnv(ae_model=ae_model)
# Create the agent
#"MlpPolicy", policy_kwargs=policy_kwargs
model = PPO(policy=ShapePPOPolicy, env=env, verbose=1, learning_rate=0.001)
model.learn(total_timesteps=40000)
model.save("ppo_shape_opt")

'''
# Create test env
env = IsoperiTestEnv(ae_model)

# Loading model
model = PPO.load("ppo_shape_opt")
model.set_env(env)

obs = env.reset()
done = False
while done == False:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    #env.render()
