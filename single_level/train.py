import torch
import torch.nn.functional as F
from torch import optim
from torch.backends import cudnn

from stable_baselines3 import SAC, PPO

from net_models import Extractor, ShapePPOPolicy
from isoperi import IsoperiEnv
from isoperi_test import IsoperiTestEnv

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from common.utils import get_samples
from common.common_nets import VAE
from common.train_utils import train_vae

# Fixed global vars
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
device = "cpu"
# Domain decomposition params
subgrid_dim = 16; latent_dim = 16; 

# Params for RL
timestep_first = 1000000; timestep_second = 1000000
learning_rate = 1e-5; log_interval = 40

# Params for VAE
vae_model = VAE(img_dim=subgrid_dim, latent_dim=latent_dim, channels=[1, 4, 8, 16])
vae_opt = optim.Adam(vae_model.parameters(), lr = 1e-3)
num_samples=int(2**15); num_vae_epochs = 10; vae_batch_size=32

# Train mode
training_sac, training_ppo = False, False
testing_sac, testing_ppo = False, False
train_vae_flag = True

## TRAINING VAE:
if train_vae_flag:
    #cudnn.benchmark=False; torch.cuda.empty_cache()
    X = get_samples(num_samples, subgrid_dim=subgrid_dim, num_milestone=5, spacing=3)
    # Sample a large "meaningful" batch of samples and then train by shuffle this minibatch
    train_vae(X, num_vae_epochs, vae_batch_size, vae_model, vae_opt, model_path='models/vae_model')
    
## TRAINING ##
if training_sac:
    pass

if training_ppo:
    # Load VAE Model
    vae_model.load_state_dict(torch.load('models/vae_model'))
    vae_model.to(device)
 
    # Make extractor (on policy common for both actor and critic nets)
    policy_kwargs = dict(
        features_extractor_class=Extractor,
        features_extractor_kwargs=dict(features_dim=16*2*2+256, 
        subgrid_dim=16, grid_dim=8, latent_dim=16),
    )
    # Create env with chosen/fixed initial shape
    env = IsoperiEnv(vae_model, grid_dim=8, subgrid_dim=16, 
                 latent_dim=16, resolution=50, shape_init='chosen')
    # Create agent and learn with fixed initial shape
    model = PPO(policy=ShapePPOPolicy, policy_kwargs=policy_kwargs, env=env, verbose=1, learning_rate=learning_rate)
    model.device = device
    model.learn(total_timesteps=timestep_first, log_interval=log_interval)
    
    # Create new env with random initial shape
    env = IsoperiEnv(vae_model, grid_dim=8, subgrid_dim=16, 
                 latent_dim=16, resolution=50, shape_init='random')
    # Agent learn with random initial shape
    model.set_env(env)
    model.learn(total_timesteps=timestep_second, log_interval=log_interval)
    
    model.save("models/ppo_shape_opt")

## TESTING ##
if testing_sac:
    pass

if testing_ppo:
    # Create test env
    env = IsoperiTestEnv(vae_model, grid_dim=8, subgrid_dim=16, 
                 latent_dim=16, resolution=50)  
    # Loading model
    model = PPO.load("models/ppo_shape_opt")
    model.set_env(env)
    
    obs = env.reset()
    done = False
    while done == False:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()