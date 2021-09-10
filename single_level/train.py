import torch
import torch.nn.functional as F
from torch import optim

from stable_baselines3 import SAC, PPO

from net_models import Extractor, ShapePPOPolicy
from isoperi import IsoperiEnv
from isoperi_test import IsoperiTestEnv

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from common.utils import get_samples
from common.common_nets import VAE

def loss_vae(x_recon, x, mu, logvar):
    ''' 
    Reconstruction + KL divergence losses summed over all elements and batch.
    See Kingma et al.
    '''
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE+KLD

# Fixed global vars
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

# Params for RL
timestep = 1000; learning_rate = 3e-4; log_interval = 1

training_sac, training_ppo = False, True
testing_sac, testing_ppo = False, False

# Params for VAE
train_vae = False
vae_model = VAE()
vae_model = VAE().to(device)
vae_opt = optim.Adam(vae_model.parameters(), lr = 1e-3)
num_samples=1000; num_vae_epoch = 10; batch_size=32

## TRAINING VAE:
if train_vae:
    # Sample a large "meaningful" batch of samples and then train by shuffle this minibatch
    X = get_samples(num_samples, subgrid_dim=16, num_milestone=5, spacing=3)
    for i_epoch in range(num_vae_epoch):
        ## shuffle
        Xtrain = X[torch.randperm(num_samples)]
        # compute total_loss for log purpose
        total_loss = 0
        num_iter = num_samples//batch_size
        for i_iter in range(num_iter):
            vae_opt.zero_grad()
            # Shuffle mini-batch
            x = Xtrain[i_iter*batch_size:(i_iter+1)*batch_size,:,:].to(device)
            # Forward and calculate loss
            mu, logvar = vae_model.encode(x)
            x_recon = vae_model.decode(vae_model.reparameterize(mu, logvar))
            #DEBUG: print(x_recon[0,0,20:25, 20:25].cpu().detach().numpy())
            loss = loss_vae(x_recon, x, mu, logvar)
            diff = F.mse_loss(x_recon, x)
            # Grad descent
            loss.backward()
            vae_opt.step()
            # Update total_loss
            total_loss += loss
            
        print('\nEpoch {}: AE batch-average loss is {}\n'.format(i_epoch+1, total_loss/num_iter))

## TRAINING ##
if training_sac:
    pass

if training_ppo:
    # Make extractor (on policy common for both actor and critic nets)
    policy_kwargs = dict(
        features_extractor_class=Extractor,
        features_extractor_kwargs=dict(features_dim=16*2*2+64),
        )
    # Create latent env
    env = IsoperiEnv(vae_model, grid_dim=8, subgrid_dim=16, 
                 latent_dim=16, resolution=50)
    # Create agent and learn
    model = PPO(policy=ShapePPOPolicy, policy_kwargs=policy_kwargs, env=env, verbose=1, learning_rate=learning_rate)
    model.learn(total_timesteps=timestep, log_interval=log_interval)
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