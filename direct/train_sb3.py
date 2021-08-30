import numpy as np
import torch

from stable_baselines3 import DDPG, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise


from sb3_model import ShapePolicy, ShapePPOPolicy
from isoperi import IsoperiEnv
from isoperi_test import IsoperiTestEnv


# Fixed global vars
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

timestep = 200000; learning_rate = 3e-4; log_interval = 100
training, training_ddpg, training_ppo = False, False, False
testing, testing_ddpg, testing_ppo = True, False, False

## TRAINING ##
if training:
    # Create latent env
    env = IsoperiEnv()
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=dict(pi=[16, 32, 16, 8, 16, 8, 4, 4], qf=[32, 16, 16, 8, 4, 8, 4]))
    model = SAC("MlpPolicy", policy_kwargs=policy_kwargs, env=env, verbose=1, learning_rate=learning_rate)
    model.device = device
    model.learn(total_timesteps=timestep, log_interval=log_interval)
    model.save("models/sac_shape_opt")
    
if training_ddpg:
    # Create latent env
    env = IsoperiEnv()
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01*np.ones(n_actions))

    #policy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                     net_arch=dict(pi=[32, 64, 64, 64, 128], qf=[32, 16, 64, 16, 64, 32]))
    #model = SAC("MlpPolicy", policy_kwargs=policy_kwargs, env=env, verbose=1) #learning_rate=0.003)
    model = DDPG(policy=ShapePolicy, action_noise=action_noise, env=env, verbose=1, learning_rate=learning_rate)
    model.device = device
    model.learn(total_timesteps=timestep, log_interval=log_interval)
    model.save("models/ddpg_shape_opt")


if training_ppo:
    # Create latent env
    env = IsoperiEnv()
    # Create agent and learn
    model = PPO(policy=ShapePPOPolicy, env=env, verbose=1, learning_rate=learning_rate)
    model.learn(total_timesteps=timestep, log_interval=log_interval)
    model.save("models/ppo_shape_opt")

## TESTING ##
if testing:
    # Create test env
    env = IsoperiTestEnv()
    
    # Loading model
    model = SAC.load("models/sac_shape_opt")
    model.set_env(env)
    
    obs = env.reset()
    done = False
    while done == False:
        action, _states = model.predict(obs)
        #action = np.random.rand(16)-.5
        obs, rewards, done, info = env.step(action)
        #env.render()
        
if testing_ddpg:
    # Create test env
    env = IsoperiTestEnv()
    
    # Loading model
    model = DDPG.load("models/ddpg_shape_opt")
    model.set_env(env)
    
    obs = env.reset()
    done = False
    while done == False:
        action, _states = model.predict(obs)
        #action = np.random.rand(16)
        obs, rewards, done, info = env.step(action)
        #env.render()

if testing_ppo:
    # Create test env
    env = IsoperiTestEnv()
    
    # Loading model
    model = PPO.load("models/ppo_shape_opt")
    model.set_env(env)
    
    obs = env.reset()
    done = False
    while done == False:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()


# Custom actor architecture with two layers of 64 units each
# Custom critic architecture with two layers of 400 and 300 units
#policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                     net_arch=dict(pi=[32, 16, 16, 8, 8, 8, 16, 16, 32], qf=[32, 16, 16, 8, 8, 8, 4, 4, 2]))

#policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                     net_arch=[64, 32, 16, 16, 16, 32, 64])