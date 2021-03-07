"""
This file consist of the training step. 
The test file will be a python notebook (TBA later)
"""

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from net_model import VAE, LatentRL
from utils import initialize, step, updateX, updateZ

# Helper types
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Hyperparams:
T = 256 # Set the number of time step in one episode T = 256
gamma = 0.05 # Discounted rate
eps = 0.01 # Helper when normalize cumulative rewards
lambda1 = 0.2 # artificial reward vs actor-critic loss

"""
initialize X, z
"""

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #Kingma et al.
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# Given previous (x, z, action) data
# We sample the next (x, z, action) data and update other values accordingly
def sample(vae_model, ac_model, X, x, p, z, action):
    # Sample latent space for VAE training
    mu, logvar = vae_model.encode(x)
    vae_model.mu_buf.append(mu)
    vae_model.logvar_buf.append(logvar)
    # Update current z
    z = update(z, p, action[16:])
    # Sample policy, net value, and artificial reward
    action, value, net_reward = ac_model(z)
    # Get quardrant position (prob) info from actions
    for i in range(4):
        m = Categorical(action[4*i, 4*i+4])
        p[i] = m.sample()
        log_probs.append(m.log_prob(p[i]))
    # Save action and artificial reward
    ac_model.save_actions.append(SavedAction(torch.stack(log_probs), value))
    ac_model.net_rewards.append(net_reward)
    # Transition
    z1 = step(z, p, action[16:])
    x = vae_model.decode(z1)
    vae_model.x_buf.append(x)
    # Get true reward
    r = updateX(X, r, p, x)
    ac_model.rewards.append(r)
    
def trainVAE_per_eps(vae_opt, vae_model):
    vae_opt.zero_grad()
    loss = loss_function(x[1:], x[:-2], vae_model.mu_buf, vae_model.logvar_buf)
    loss.backward()
    # Log the loss here
    # To be filled
    vae_opt.step()
    del vae_model.x[:-2]
    del vae_model.mu_buf[:]
    del vae_model.logvar_buf[:]
    
def trainAC_per_eps(ac_opt, ac_model):
    policy_losses = [] #list to save actor loss
    value_losses = [] #list to save critic loss
    
    #Calculate cumulative reward R
    cum_rewards = []
    R = 0
    for r in ac_model.rewards[:]:
        R = r + gamma*R
        cum_rewards.insert(0, R)
    
    # Tensorize and normalize cumulative rewards
    cum_rewards = torch.tensor(cum_rewards)
    cum_rewards = (cum_rewards-cum_rewards.mean())/(cum_rewards.std() + eps)
    
    for (log_prob, value), R in zip(ac_model.saved_actions, cum_rewards):
        advantage = R - value.item()
        # calculate policy losses
        # assume that action only random around the choice of quadrant
        policy_losses.append(-log_prob*advantage)
        # calculate critic losses
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    
    ac_opt.zero_grad()
    # sum up all loss
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            + 2*torch.log(torch.tensor(np.array(ac_model.rewards) 
                - np.array(ac.model.net_reward))).sum()
    loss.backward()
    ac_opt.step()
    del model.rewards[:-2]
    del models.saved_actions[:]
    del.model.net_rewards[:-2]
        
def training_per_eps(vae_opt, vae_model, ac_opt, ac_model, X):
""" Sampling, then train VAE, then train actor critic for one episode.
Then refresh.
ac_model: RL net (actor+policy+reward net)
X: is the big image updated in small parts overtime
"""
    for t in range(T):
        sample(vae_model, ac_model, X)
    trainVAE_per_eps(vae_opt, vae_model)
    trainAC_per_eps(ac_opt, ac_model)

def train():
    # Build models and corresponding optimizers
    vae_model = VAE()
    vae_opt = optim.Adam(vae_model.parameters(), lr = 3e-2)
    ac_model = LatentRL()
    ac_opt = optim.Adam(vae_model.parameters(), lr = 3e-2)
    # Training
    for i_episode in num_episode:
        if i_episode % 10 == 0:
            initialize(x)
        training_per_eps(vae_opt, vae_model, ac_opt, ac_model, X)
        # Log results
        # To be filled
    

    




