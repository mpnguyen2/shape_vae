import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical, multivariate_normal
from torch.nn import functional as F
from collections import namedtuple
import time

from common.utils import pertube
from common.common_nets import VAE

# Variables get updated every timestep: X, area, peri
# X is refreshed every initialize_interval episodes

# Helper types
SavedAction = namedtuple('SavedAction', ['p_logprob', 'v_logprob', 'value'])

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
device = "cpu"
'''
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
'''

def loss_bce(x, x_recon):
    return F.binary_cross_entropy(x_recon, x, reduction='sum')

def loss_vae(x_recon, x, mu, logvar):
    ''' 
    Reconstruction + KL divergence losses summed over all elements and batch.
    See Kingma et al.
    '''
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE+KLD

def train_vae(Xdata, num_epochs, batch_size, vae_model, vae_opt, model_path):
    '''
    Training a VAE
    Args:
        Xdata (torch.Tensor): whole dataset
        num_epochs (int): number of epochs over the whole dataset Xdata
        batch_size (int): training batch size
        vae_model (nn.Module): VAE net model
        vae_opt: optimizer for VAE
        model_path (str): path for saving model
    '''
    # Find number of samples
    num_samples = Xdata.shape[0]
    ## Training the VAE for num_epochs epochs
    for i_epoch in range(num_epochs):
        # Shuffle dataset each epoch
        Xtrain = Xdata[torch.randperm(num_samples)].to(device)
        # compute total_loss for log purpose
        total_loss = 0
        num_iter = num_samples//batch_size
        for i_iter in range(num_iter):
            vae_opt.zero_grad()
            # Take batch data x
            x = Xtrain[i_iter*batch_size:(i_iter+1)*batch_size,:,:].to(device)
            # Forward and calculate loss
            mu, logvar = vae_model.encode(x)
            x_recon = vae_model.decode(vae_model.reparameterize(mu, logvar))
            loss = loss_vae(x_recon, x, mu, logvar)
            # Grad descent
            loss.backward()
            vae_opt.step()
            # Update total_loss
            total_loss += loss            
        #del Xtrain; torch.cuda.empty_cache() # CUDA unload memory TBA
        # Print log
        print('Epoch {}: VAE batch-average loss is {}\n'.format(i_epoch+1, total_loss/num_iter))
    
    # Save model
    torch.save(vae_model.state_dict(), model_path)


def train_lae(Xdata, num_epochs, batch_size, ae_model, ae_opt, model_path, alpha=5):
    '''
    Training a Lipschitz autoencoder
    Args:
        Xdata (torch.Tensor): whole dataset
        num_epochs (int): number of epochs over the whole dataset Xdata
        batch_size (int): training batch size
        vae_model (nn.Module): VAE net model
        vae_opt: optimizer for VAE
        model_path (str): path for saving model
    '''
    num_samples = Xdata.shape[0]
    for i_epoch in range(num_epochs):
        # Shuffle dataset each epoch
        Xtrain = Xdata[torch.randperm(num_samples)].to(device)
        # compute total_loss for log purpose
        total_loss = 0
        num_iter = num_samples//batch_size
        for i_iter in range(num_iter):
            ae_opt.zero_grad()
            # Take batch data x
            x = Xtrain[i_iter*batch_size:(i_iter+1)*batch_size,:,:].to(device)
            xp = pertube(x)
            # Forward and calculate loss
            x_recon, z = ae_model(x)
            xp_recon, zp = ae_model(xp)
            
            loss = loss_bce(x_recon, x) 
            loss += loss_bce(xp_recon, xp)
            loss += alpha*F.smooth_l1_loss(z, zp, reduction='sum', beta=0.01)

            # Grad descent
            loss.backward()
            ae_opt.step()
            # total loss for logging
            total_loss += loss
        
        # Print log
        print('Epoch {}: AE batch-average loss is {}\n'.format(i_epoch+1, total_loss/num_iter))
        
    # Save model
    torch.save(ae_model.state_dict(), model_path)
    
def sample(vae_model, ac_model, X, z, area, peri):
    ''' 
    Sample-efficient for a single hierarchy level. TBA    
    '''
    
    # Reconstruct new x from new z and put it to x reconstruction buffer
    #vae_model.x_recon_buf.append(vae_model.decode(z1.unsqueeze(0)).squeeze(0))
    
    # Sample policy, value fct, and artificial reward
    action, value, net_reward = ac_model(z)
   
    '''
    # Get quardrant position (log prob) info from actions
    b = z.shape[0]
    p_logprobs = torch.zeros(b, 4)
    for b1 in range(b):
        for i in range(4):
            m = Categorical(action[b1, 4*i:4*i+4])
            tmp = m.sample()
            p[b1, i] = int(tmp)
            p_logprobs[b1, i] = m.log_prob(tmp)
    # Get logprob of the amount of change v, which is modeled as a Gaussian with fixed variance
    v = torch.zeros(b, 8).to(device)
    v_logprobs = []
    for b1 in range(b):
        m = multivariate_normal.MultivariateNormal(loc=action[b1, 16:], covariance_matrix=0.05*torch.eye(8).to(device))
        v[b1] = m.sample()
        v_logprobs.append(m.log_prob(v[b1]))
    # Save action (log_prob) and artificial reward
    ac_model.saved_actions.append(SavedAction(p_logprobs.to(device), torch.stack(v_logprobs).to(device), value))
    ac_model.net_rewards.append(net_reward)
    '''
    return
    
def train(num_eps, initialize_interval):
    """
    Training: Train both VAE and actor-critic-artificial reward if needed
    Sample and train at the same time. Keep sample with global X and Z so that 
    sampling process is efficient
    """
    # Build vae model and corresponding optimizer
    vae_model = VAE().to(device)
    vae_opt = optim.Adam(vae_model.parameters(), lr = 1e-2)
    # Build ac model
    ac_model = nn.Module().to(device)
    # Build ac optimizer
    ac_opt = optim.Adam(ac_model.parameters(), lr = 1e-3)

    # Training
    print('Begin training...\n')
    start_time = time.time()

    # Pretrain the VAE model   
    print('Pretrain VAE:\n')
    cur_time = start_time
    #train_vae(Xdata, num_epochs, batch_size, vae_model, vae_opt, model_path)
    print('\nFinish pretrainning for VAE. This takes {} seconds\n'.format(time.time()-cur_time))
    
    # Global variables
    X = None; Z = None
    
    cur_time = time.time()
    for i_episode in range(num_eps):
        if i_episode % initialize_interval == 0:
            # Perform new intialization
            # Sample efficnetly and training actor-critic one episode
            pass
   
    return