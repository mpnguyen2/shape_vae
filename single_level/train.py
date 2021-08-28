# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:34:32 2021

@author: nguyn
"""

import numpy as np
import torch
from torch import optim
from torch.distributions import multivariate_normal
from torch.nn import functional as F
from collections import namedtuple
import time

from net_model import VAE, LatentRL
from utils import get_batch_sample, initialize, step, reward


# Variables get updated every timestep: X, area, peri
# X is refreshed every initialize_interval episodes

# Helper types
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Fixed hyperparams:
beta = 0.00001        
eps = 0.01 # Helper when normalize cumulative rewards
lambda1 = 0.1 # artificial reward vs actor-critic loss
lambda2 = 0.2 # helper hyperparam so that binary cross entropy can be calculated on 0-1 grid
# GPU to use
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#device = torch.device("cpu")
def loss_function(x_recon, x, mu, logvar):
    ''' 
    Reconstruction + KL divergence losses summed over all elements and batch.
    See Kingma et al.
    '''
    BCE = F.binary_cross_entropy(x_recon, x, reduction="mean")
    KLD = beta * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    return BCE+KLD

def pretrain_vae(vae_model, vae_opt, num_iter, sample_size=512):
    num_epoch = 1001; #total_loss = 0
    for i_iter in range(num_iter):
        # Sample a mini-batch and then train by shuffle this minibatch
        X = get_batch_sample(sample_size, device) # 1024 x 32 x 32
        # compute total_loss for log purpose
        #total_loss = 0
                
        for i_epoch in range(num_epoch):
            vae_opt.zero_grad()
            # Shuffle mini-batch
            x = X[torch.randperm(sample_size)[:32],:,:]
            x = x.to(device)
            # Forward and calculate loss
            mu, logvar = vae_model.encode(x)
            x_recon = vae_model.decode(vae_model.reparameterize(mu, logvar))
            
            loss = loss_function(x_recon, x, mu, logvar)
            #diff = F.mse_loss(x_recon, x)
            if i_epoch % 1000 == 0:
                print('VAE loss in {}th epoch of the {}th iteration is {:.5f}'\
                      .format(i_epoch+1, i_iter+1, loss.item()))
            # Grad descent
            loss.backward()
            vae_opt.step()
            # Update total_loss
            #total_loss += loss
        #x = X[torch.randperm(sample_size)[:32],:,:,:]
        #visualize_grid(vae_model, x, output="./pretrain/image")
    print("Finished Pre Train")

        
def sample(vae_model, ac_model):
    ''' 
    Given previous (x, z, action) data
    We sample the next (x, z, action) data and update other values accordingly
    '''
    x = vae_model.x_buf[-1]
    # Sample data for VAE training
    #mu, logvar = vae_model.encode(x)
    #vae_model.mu_buf.append(mu.squeeze())
    #vae_model.logvar_buf.append(logvar.squeeze())
    
    # Encode x by z
    mu, logvar = vae_model.encode(x.unsqueeze(0))
    z = vae_model.reparameterize(mu, logvar).squeeze(0)
    
    # Reconstruct new x from new z and put it to x reconstruction buffer
    #vae_model.x_recon_buf.append(vae_model.decode(z1.unsqueeze(0)).squeeze(0))
    
    # Sample policy, value fct, and artificial reward
    act_mean, act_logvar, value = ac_model(z.unsqueeze(0))
    act_mean, act_var, value = act_mean.squeeze(0), \
        torch.diag(torch.exp(act_logvar.squeeze(0))), value.squeeze(0)
    
    # Get how much change v from actions (modeled as a Gaussian)
    m = multivariate_normal.MultivariateNormal(loc=act_mean, covariance_matrix=act_var)
    v = m.sample()
    
    # Step into new z
    z = step(z, v, device)
    
    # Save action (log_prob) and artificial reward
    ac_model.saved_actions.append(SavedAction(m.log_prob(v), value))
    #ac_model.net_rewards.append(net_reward)

    
    # x is updated so that x can be fed into the next iteration
    x = vae_model.decode(z.unsqueeze(0)).squeeze(0)
    # x should only contain 0 and 1
    x = torch.round(x)
    vae_model.x_buf[-1] = x
    
    ac_model.rewards.append(reward(x.cpu().detach().numpy()))
    
def trainVAE_per_eps(vae_opt, vae_model):
    """
    Train the VAE net for one episode. Take mini-batch data as the entire
    (x, x_recon, z1) from time to time
    """
    vae_opt.zero_grad()
    loss = loss_function(torch.stack(vae_model.x_recon_buf), \
            torch.stack(vae_model.x_buf[:-1]), \
            torch.stack(vae_model.mu_buf), \
                torch.stack(vae_model.logvar_buf))
    # Grad descent
    loss.backward()
    vae_opt.step()
    
    # Clean buffer for the next episode. 
    # Keep only one last x to feed to the next episode
    del vae_model.x_buf[:-1]
    del vae_model.x_recon_buf[:]
    del vae_model.mu_buf[:]
    del vae_model.logvar_buf[:]
    return loss
    
def trainAC_per_eps(ac_opt, ac_model, gamma):
    """
    Train the actor-critic-artificial reward net for one episode
    """
    policy_losses = [] #list to save actor loss
    #value_losses = [] #list to save critic loss
    
    # Calculate cumulative reward R backwardly
    cum_rewards = []
    R = 0
    for r in ac_model.rewards[:]:
        R = float(r + gamma*R)
        cum_rewards.insert(0, R)
    
    # Tensorize and normalize cumulative rewards
    cum_rewards = torch.tensor(cum_rewards).to(device)
    cum_rewards = (cum_rewards-cum_rewards.mean())/(cum_rewards.std() + eps)
    
    # Calculate individual terms in policy and critic loss fcts
    for (log_prob, value), R in zip(ac_model.saved_actions, cum_rewards):
        advantage = R #- value.item()
        # calculate policy losses
        # assume that action only random around the choice of quadrant
        policy_losses.append(-log_prob*advantage)
        # calculate critic losses
        #value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))
    
    ac_opt.zero_grad()
    # sum up all loss
    loss = torch.stack(policy_losses).sum() #+ torch.stack(value_losses).sum()

    # grad descent
    loss.backward()
    '''
    # DEBUGGING
    print('\nPolicy loss: {}'.format(loss))
    for name, param in ac_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if name=='p_net.6.weight' or name=='v_net.6.weight'\
                or name=='p_net.6.bias' or name=='v_net.6.bias':
                print(name, torch.abs(param.grad).sum())
    '''
    ac_opt.step()
    
    last_reward = ac_model.rewards[-1]
    del ac_model.rewards[:]
    del ac_model.saved_actions[:]
    
    return last_reward
        
def training_per_eps(vae_opt, vae_model, ac_opt, ac_model, T, gamma):
    """ 
    Sampling, then train VAE, then train actor critic for one episode.
    Then refresh.
    ac_model: RL net (actor+policy+reward net)
    X: is the big image updated in small parts overtime
    """

    for t in range(T):
        sample(vae_model, ac_model)
    #vae_loss = trainVAE_per_eps(vae_opt, vae_model)
    last_reward = trainAC_per_eps(ac_opt, ac_model, gamma)
    return last_reward

def training_per_eps_without_vae(ac_opt, ac_model, T, gamma):
    pass
    
def train(num_iter_pretrain_vae, num_eps_with_VAE, \
          initialize_interval, T, discounted_rate, \
              log_interval, save_interval, learning_rate = 3e-2):
    """
    Training: Train both VAE and actor-critic-artificial reward for num_eps1 episodes
    Then remove the VAE, and only train the actor-critic for another num_eps2 episodes
    """
    # Build models and corresponding optimizers
    vae_model = VAE().to(device)
    vae_opt = optim.Adam(vae_model.parameters(), lr = learning_rate)
    ac_model = LatentRL().to(device)
    ac_opt = optim.Adam(ac_model.parameters(), lr = learning_rate)
    
    # Training
    print('Begin training...\n')
    start_time = time.time()

    # Pretrain the VAE model   
    print('Pretrain VAE:\n')
    cur_time = start_time
    pretrain_vae(vae_model, vae_opt, num_iter_pretrain_vae)
    torch.save(vae_model.state_dict(), 'models/vae_model1')
    print('\nFinish pretrainning for VAE. This takes {:.3f} seconds'.format(time.time()-cur_time))
    
    # Initialize x
    x = torch.zeros(32, 32)
    
    rewards = [] # rewards log
    
    ## TRAIN WITH VAE ##
    cur_time = time.time()
    for i_episode in range(num_eps_with_VAE):
        if i_episode % initialize_interval == 0:
            # Initialize X and z every 10 episodes
            area, peri, x = initialize(device)
            init_reward = np.sqrt(area)/peri
            # Push new initialize x to vae model x buffer
            if len(vae_model.x_buf) == 0:
                vae_model.x_buf.append(x)
            else:
                vae_model.x_buf[0] = x
        # training one episode
        reward = training_per_eps(vae_opt, vae_model, \
                        ac_opt, ac_model, T, discounted_rate) 
        rewards.append(reward)
        # Log result every log_interval episodes
        if i_episode % log_interval == 0:
            # Log VAE loss and current cumulative reward
            print('Episode {}: init_reward:{:.3f}, cumulative reward: {:.3f}; percentage: {:.2f}; Time elapsed: {:.3f} seconds\n'\
                  .format(i_episode+1, init_reward, reward, 100*reward/init_reward, time.time()-cur_time))
            cur_time = time.time()
        if i_episode % save_interval == 0:
            torch.save(ac_model.state_dict(),'models/ac_model1')
            
    print('\nTraining with VAE completed. Time taken: {} seconds\n'.format(time.time()-start_time))
    '''
    print('AC Model param')
    for param in ac_model.parameters():
        print(param.data)
    '''
    ## TRAIN WITHOUT VAE: ADDED LATER ##
    '''
    start_time = time.time()
    cur_time = time.time()
    for i_episode in range(num_eps_without_VAE):
        reward = training_per_eps_without_vae(ac_opt, ac_model, T, discounted_rate)
    print('\nTraining completed. Time taken in total: {} seconds\n'.format(time.time()-start_time))
    '''
    return rewards