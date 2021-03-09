"""
This file consist of the training step. 
The test file is in a python notebook.
"""

import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical
from torch.nn import functional as F
from net_model import VAE, LatentRL
from collections import namedtuple
from utils import initialize, step, updateX, updateZ, decode_pic

# Helper types
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Hyperparams:
T = 10 # Set the number of time step in one episode T = 256
gamma = 0.05 # Discounted rate
eps = 0.01 # Helper when normalize cumulative rewards
lambda1 = 5 # artificial reward vs actor-critic loss
num_eps1 = 6
num_eps2 = 1

# Variables get updated every timestep until every 10 episodes are:
# X, area, peri
 
def loss_function(recon_x, x, mu, logvar):
    ''' 
    Reconstruction + KL divergence losses summed over all elements and batch.
    See Kingma et al.
    '''
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def sample(vae_model, ac_model, p, z, X, area, peri):
    ''' 
    Given previous (x, z, action) data
    We sample the next (x, z, action) data and update other values accordingly
    '''
    x = vae_model.x_buf[0]
    # Sample data for VAE training
    mu, logvar = vae_model.encode(x.unsqueeze(0))
    vae_model.mu_buf.append(mu.squeeze())
    vae_model.logvar_buf.append(logvar.squeeze())
    
    # Sampe z and z1 from the VAE. Note that in the previous iteration, 
    # z1 is updated when it hasn't passed through decoder yet.
    # We have the new x and the new z and z1 will be determined by 
    # this new x instead of the transition step
    z1 = vae_model.reparameterize(mu, logvar).squeeze()
    z = updateZ(z, p, z1)
    
    # Reconstruct new x from new z and put it to x reconstruction buffer
    vae_model.x_recon_buf.append(vae_model.decode(z1.unsqueeze(0)).squeeze(0))
    
    # Sample policy, value fct, and artificial reward
    action, value, net_reward = ac_model(z.unsqueeze(0))
    action, value, net_reward = action.squeeze(0), value.squeeze(0), net_reward.squeeze(0)
    
    # Get quardrant position (prob) info from actions
    log_probs = []
    for i in range(4):
        m = Categorical(action[4*i:4*i+4])
        p[i] = int(m.sample())
        log_probs.append(m.log_prob(torch.tensor(p[i])))
    # Save action (log_prob) and artificial reward
    ac_model.saved_actions.append(SavedAction(torch.stack(log_probs), value))
    ac_model.net_rewards.append(net_reward)
    
    # Transition to next z1 given current z, modifying position of z, and how much to add
    # x is updated so that x can be fed into the next iteration
    z1 = step(z, p, action[16:])
    x = vae_model.decode(z1.unsqueeze(0))
    vae_model.x_buf.append(x.squeeze(0).clone().detach())
    
    # Get true reward
    area, peri = updateX(X, area, peri, p, x)
    r = float(np.sqrt(area)/peri)
    #print('Area: {}, Peri: {}, reward: {}'.format(area, peri, r))
    ac_model.rewards.append(r)
    return area, peri
    
def trainVAE_per_eps(vae_opt, vae_model):
    """
    Train the VAE net for one episode. Take mini-batch data as the entire
    (x, x_recon, z1) from time to time
    """
    vae_opt.zero_grad()
    loss = loss_function(torch.stack(vae_model.x_recon_buf), \
            torch.stack(vae_model.x_buf[:-1]), \
            torch.stack(vae_model.mu_buf), torch.stack(vae_model.logvar_buf))
    loss.backward()
    # Log the loss here
    # To be filled
    vae_opt.step()
    del vae_model.x_buf[:-1]
    del vae_model.x_recon_buf[:]
    del vae_model.mu_buf[:]
    del vae_model.logvar_buf[:]
    return loss
    
def trainAC_per_eps(ac_opt, ac_model):
    """
    Train the actor-critic-artificial reward net for one episode
    """
    policy_losses = [] #list to save actor loss
    value_losses = [] #list to save critic loss
    
    # Calculate cumulative reward R backwardly
    cum_rewards = []
    R = 0
    for r in ac_model.rewards[:]:
        R = float(r + gamma*R)
        cum_rewards.insert(0, R)
    
    # Tensorize and normalize cumulative rewards
    cum_rewards = torch.tensor(cum_rewards)
    cum_rewards = (cum_rewards-cum_rewards.mean())/(cum_rewards.std() + eps)
    
    # Calculate individual terms in policy and critic loss fcts
    for (log_prob, value), r in zip(ac_model.saved_actions, cum_rewards):
        advantage = r - value.item()
        # calculate policy losses
        # assume that action only random around the choice of quadrant
        policy_losses.append(-log_prob*advantage)
        # calculate critic losses
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    
    ac_opt.zero_grad()
    # sum up all loss
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()\
        + lambda1*torch.log(F.mse_loss(torch.tensor(ac_model.rewards), \
                torch.stack(ac_model.net_rewards).squeeze()))
    # grad descent
    loss.backward()
    ac_opt.step()
    # log cumulative reward
    print('The cumulative reward in the episode is: {}\n'.format(R))
    
    del ac_model.rewards[:]
    del ac_model.saved_actions[:]
    del ac_model.net_rewards[:]
        
def training_per_eps(vae_opt, vae_model, ac_opt, ac_model,\
                     p, z, X, area, peri):
    """ 
    Sampling, then train VAE, then train actor critic for one episode.
    Then refresh.
    ac_model: RL net (actor+policy+reward net)
    X: is the big image updated in small parts overtime
    """

    for t in range(T):
        area, peri = sample(vae_model, ac_model, p, z, X, area, peri)
    VAE_loss = trainVAE_per_eps(vae_opt, vae_model)
    # Log VAE loss
    print('VAE loss in this episode is: {}'.format(VAE_loss))
    trainAC_per_eps(ac_opt, ac_model)
    return area, peri

def training_per_eps_without_vae(ac_opt, ac_model):
    # Initialize p and z
    p = np.zeros(4).astype(int)
    z = torch.rand(8, 16, 16)
    ## Sample 1 trajector ##
    for t in range(T):
        # Sample policy, value fct, and artificial reward
        action, value, net_reward = ac_model(z.unsqueeze(0))
        action, value, net_reward = action.squeeze(0), value.squeeze(0), net_reward.squeeze(0)
        # Get quardrant position (prob) info from actions
        log_probs = []
        for i in range(4):
            m = Categorical(action[4*i: 4*i+4])
            p[i] = int(m.sample())
            log_probs.append(m.log_prob(torch.tensor(p[i])))
        # Save action (log_prob) and artificial reward
        ac_model.saved_actions.append(SavedAction(torch.stack(log_probs), value))
        ac_model.net_rewards.append(net_reward)
        # Transition to next z1 given current z, and then update z
        z1 = step(z, p, action[16:])
        z = updateZ(z, p, z1)
    ## Optimize ac_net ##
    policy_losses = [] #list to save actor loss
    value_losses = [] #list to save critic loss
    
    # Calculate cumulative reward R backwardly
    cum_rewards = []
    R = 0
    for r in ac_model.net_rewards[:]:
        R = r.detach().numpy() + gamma*R
        cum_rewards.insert(0, R)
        
    # Tensorize and normalize cumulative rewards
    cum_rewards = torch.tensor(cum_rewards)
    cum_rewards = (cum_rewards-cum_rewards.mean())/(cum_rewards.std() + eps)
    
    # Calculate individual terms in policy and critic loss fcts
    for (log_prob, value), r in zip(ac_model.saved_actions, cum_rewards):
        advantage = r - value.item()
        # calculate policy losses
        # assume that action only random around the choice of quadrant
        policy_losses.append(-log_prob*advantage)
        # calculate critic losses
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    
    ac_opt.zero_grad()
    # sum up all loss
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    # grad descent
    loss.backward()
    ac_opt.step()
    # log cumulative reward
    print('The cumulative reward in the episode is: {}\n'.format(R))
    
    del ac_model.saved_actions[:]
    del ac_model.net_rewards[:]
    
def train():
    """
    Training: Train both VAE and actor-critic-artificial reward for num_eps1 episodes
    Then remove the VAE, and only train the actor-critic for another num_eps2 episodes
    """
    # Build models and corresponding optimizers
    vae_model = VAE()
    vae_opt = optim.Adam(vae_model.parameters(), lr = 3e-2)
    ac_model = LatentRL()
    ac_opt = optim.Adam(vae_model.parameters(), lr = 3e-2)
    # Declare the big image X
    # X get a small part updated each time step and refresh every 10 episodes
    X = np.zeros((512, 512))
    # Training
    for i_episode in range(num_eps1):
        print('Episode {}:'.format(i_episode+1))
        if i_episode % 10 == 0:
            # Initialize X and z every 10 episodes
            area, peri = initialize(X)
            # Set each (8, :, :,) of z to be the encoding of a cell in the subgrid of X
            z = decode_pic(X, vae_model)
            p = np.zeros(4).astype(int)
            if len(vae_model.x_buf) == 0:
                vae_model.x_buf.append(torch.tensor(X[0:32, 0:32], dtype=torch.float).unsqueeze(0))
            else:
                vae_model.x_buf[0] = torch.tensor(X[0:32, 0:32], dtype=torch.float).unsqueeze(0)
        # training one episode
        area, peri = training_per_eps(vae_opt, vae_model, \
                        ac_opt, ac_model, p, z, X, area, peri) 
    print('\nTraining with VAE complete.\n')
    
    for i_episode in range(num_eps2):
        training_per_eps_without_vae(ac_opt, ac_model)
    print('Training completed.')
    
train()