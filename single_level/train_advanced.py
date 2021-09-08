import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical, multivariate_normal
from torch.nn import functional as F
from net_model_2 import VAE, LatentRL
from collections import namedtuple

import time

from utils import initialize, step, updateX, updateZ, decode_pic, get_batch_sample

# Variables get updated every timestep: X, area, peri
# X is refreshed every initialize_interval episodes

# Helper types
SavedAction = namedtuple('SavedAction', ['p_logprob', 'v_logprob', 'value'])

# Fixed hyperparams:
eps = 0.0001 # Helper when normalize cumulative rewards
lambda1 = 0.1 # artificial reward vs actor-critic loss
lambda2 = 0.2 # helper hyperparam so that binary cross entropy can be calculated on 0-1 grid
# GPU to use
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

'''
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
'''

def loss_function(x_recon, x, mu, logvar):
    ''' 
    Reconstruction + KL divergence losses summed over all elements and batch.
    See Kingma et al.
    '''
    BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE+KLD

def pretrain_vae(vae_model, vae_opt, training_size, num_epoch, num_train_iter=1):
    for i_iter in range(num_train_iter):
        # Sample a mini-batch and then train by shuffle this minibatch
        X = get_batch_sample(training_size) 
        # compute total_loss for log purpose
        #total_loss = 0
        for i_epoch in range(num_epoch):
            vae_opt.zero_grad()
            # Shuffle mini-batch
            x = X[torch.randperm(training_size)[:32],:,:]
            x = x.to(device)
            # Forward and calculate loss
            mu, logvar = vae_model.encode(x)
            x_recon = vae_model.decode(vae_model.reparameterize(mu, logvar))
            
            #print(x_recon[0,0,20:25, 20:25].cpu().detach().numpy())
            loss = loss_function(x_recon, x, mu, logvar)
            diff = F.mse_loss(x_recon, x)
            if i_epoch % 1000 == 0:
                print('VAE loss in {}th epoch of the {}th iteration is {}'\
                      .format(i_epoch+1, i_iter+1, diff.item()))
            # Grad descent
            loss.backward()
            vae_opt.step()
            
            # Update total_loss
            #total_loss += loss
        #print('\nIteration {}: VAE average loss: {}\n'.format(i_iter+1, total_loss/num_epoch))

def sample(vae_model, ac_model, p, z, X, X1, area, peri):
    ''' 
    Given previous (x, z, action) data
    We sample the next (x, z, action) data and update other values accordingly
    '''
    x = vae_model.x_buf[-1]
    # Sample data for VAE training
    mu, logvar = vae_model.encode(x)
    #vae_model.mu_buf.append(mu.squeeze())
    #vae_model.logvar_buf.append(logvar.squeeze())
    
    # Sampe z and z1 from the VAE. Note that in the previous iteration, 
    # z1 is updated when it hasn't passed through decoder yet.
    # We have the new x and the new z and z1 will be determined by 
    # this new x instead of the transition step
    z1 = vae_model.reparameterize(mu, logvar)
    z = updateZ(z, p, z1, device)
    
    # Reconstruct new x from new z and put it to x reconstruction buffer
    #vae_model.x_recon_buf.append(vae_model.decode(z1.unsqueeze(0)).squeeze(0))
    
    # Sample policy, value fct, and artificial reward
    action, value, net_reward = ac_model(z)
   
    #print(value)
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

    # Transition to next z1 given current z, modifying position of z, and how much to add
    # x is updated so that x can be fed into the next iteration
    z1 = step(z, p, v, device)
    x = vae_model.decode(z1)
    #vae_model.x_buf.append(x.squeeze(0).clone().detach())
    
    # Get true reward
    area, peri = updateX(X, X1, area, peri, p, x)
    ac_model.rewards.append((np.sqrt(area)/np.maximum(1, peri)).astype(float))
    
    return area, peri
    
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
    
def trainAC_per_eps(ac_opt, ac_model, gamma, T, episode, b):
    """
    Train the actor-critic-artificial reward net for one episode
    """
    p_policy_losses = [] #list to save actor loss for position
    v_policy_losses = [] #list to save actor loss for the change amount v
    value_losses = [] #list to save critic loss
    
    # Calculate cumulative reward R backwardly
    cum_rewards = []
    R = np.zeros(b)
    for r in ac_model.rewards[:]:
        R = r + gamma*R
        cum_rewards.insert(0, R)
    
    # Tensorize and normalize cumulative rewards
    cum_rewards = torch.tensor(cum_rewards).to(device)
    #cum_rewards = (cum_rewards-cum_rewards.mean(dim=0))/(cum_rewards.std(dim=0) + eps)
    
    # Calculate individual terms in policy and critic loss fcts
    for (p_logprob, v_logprob, value), R in zip(ac_model.saved_actions, cum_rewards):
        advantage = R.unsqueeze(1) # if actor have to subtract value.item()
        # calculate policy losses
        #print(-v_logprob*advantage)
        p_policy_losses.append(-p_logprob*advantage)
        v_policy_losses.append(-v_logprob*advantage)
        # calculate critic losses
        # value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        
    ac_opt.zero_grad()
    
    # sum up all loss
    p_loss = torch.stack(p_policy_losses).sum()
    v_loss = torch.stack(v_policy_losses).sum()
    loss = p_loss + v_loss

    #value_losses = torch.stack(value_losses).mean()
    # Discourage similar values to escape local min
    #loss -= torch.log(F.mse_loss(torch.stack(values[:-1]), torch.stack(values[1:])) + eps).sum()
    
    #reward_loss = F.mse_loss(torch.tensor(ac_model.rewards),\
    #            torch.stack(ac_model.net_rewards).squeeze(), reduction='mean')
    reward_loss = 0
    
    # grad descent
    loss.backward()

    # DEBUGGING
    if episode%2 == 0:
        print('\nP loss: {}; V loss: {}; Policy loss: {}'.format(p_loss, v_loss, loss))
        for name, param in ac_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name=='p_net.8.weight' or name=='v_net.6.weight'\
                    or name=='p_net.8.bias' or name=='v_net.6.bias':
                    print(name, torch.abs(param.grad).sum())
        
    ac_opt.step()
    
  
    log_reward = ac_model.rewards[-1].mean().item()
    #print(loss)
    #print(100*torch.stack(value_losses).sum().item())
    
    del ac_model.rewards[:]
    del ac_model.saved_actions[:]
    del ac_model.net_rewards[:]
    
    return log_reward, reward_loss
        
def training_per_eps(vae_opt, vae_model, ac_opt, ac_model,\
                     p, z, X, X1, area, peri, T, gamma, episode):
    """ 
    Sampling, then train VAE, then train actor critic for one episode.
    Then refresh.
    ac_model: RL net (actor+policy+reward net)
    X: is the big image updated in small parts overtime
    """
    b = X.shape[0]
    for t in range(T):
        area, peri = sample(vae_model, ac_model, p, z, X, X1, area, peri)
    vae_loss = 0 #trainVAE_per_eps(vae_opt, vae_model)
    reward, reward_loss = trainAC_per_eps(ac_opt, ac_model, gamma, T, episode, b)
    return area, peri, vae_loss, reward, reward_loss

def training_per_eps_without_vae(ac_opt, ac_model, T, gamma):
    pass
    
def train(vae_training_size, vae_num_epoch, num_eps_with_VAE,\
          num_eps_without_VAE, num_traj, initialize_interval, T,\
          discounted_rate, log_interval):
    """
    Training: Train both VAE and actor-critic-artificial reward for num_eps1 episodes
    Then remove the VAE, and only train the actor-critic for another num_eps2 episodes
    """
    # Build vae model and corresponding optimizer
    vae_model = VAE().to(device)
    vae_opt = optim.Adam(vae_model.parameters(), lr = 1e-2)
    # Build ac model
    ac_model = LatentRL().to(device)
    # Specify different learning rates for different group of layers' params
    #special_list = ['v_net.6.weight', 'v_net.6.bias']
    #special_params = list(filter(lambda kv: kv[0] in special_list, ac_model.named_parameters()))
    #base_params = list(filter(lambda kv: kv[0] not in special_list, ac_model.named_parameters()))
    # Build ac optimizer
    ac_opt = optim.Adam(
    [
        {"params": ac_model.v_net.parameters(), "lr": 1e-3}
    ],
    lr=1e-1)
    # Training
    print('Begin training...\n')
    start_time = time.time()

    # Pretrain the VAE model   
    print('Pretrain VAE:\n')
    cur_time = start_time
    pretrain_vae(vae_model, vae_opt,  vae_training_size, vae_num_epoch)
    print('\nFinish pretrainning for VAE. This takes {} seconds\n'.format(time.time()-cur_time))
    
    # Declare data for main training
    # Large image X get a small part updated each time step and refresh every 10 episodes
    X = np.zeros((num_traj, 512, 512));
    rewards = [] # rewards log
    
    ## TRAIN WITH VAE ##
    cur_time = time.time()
    for i_episode in range(num_eps_with_VAE):
        if i_episode % initialize_interval == 0:
            print('\nNew initialization...\n')
            # Initialize X and z every 10 episodes
            area, peri, X, X1 = initialize(num_traj) 
            init_reward = np.sqrt(area)/np.maximum(peri,1)
            # Set each (8, :, :,) of z to be the encoding of a cell in the subgrid of X
            z = decode_pic(X1, vae_model, device)
            #print(torch.abs(z).sum())
            p = np.zeros((num_traj, 4)).astype(int)
            tmp = torch.tensor(X1[:, 0:32, 0:32], dtype=torch.float).to(device)
            if len(vae_model.x_buf) == 0:
                vae_model.x_buf.append(tmp)
            else:
                vae_model.x_buf[0] = tmp
        # training one episode
        area, peri, vae_loss, log_reward, reward_loss = training_per_eps(vae_opt, vae_model, \
                        ac_opt, ac_model, p, z, X, X1, area, peri, T, discounted_rate, i_episode) 
        # Log result every log_interval episodes
        if i_episode % log_interval == 0:
            # Log VAE loss and current cumulative reward
            print('\nEpisode {}: VAE loss: {}; init_reward:{}, cumulative reward: {}; Time elapsed: {} seconds\n'\
                  .format(i_episode+1, vae_loss, init_reward.mean().item(), log_reward, time.time()-cur_time))
            rewards.append(log_reward)
            cur_time = time.time()
    print('\nTraining with VAE completed. Time taken: {} seconds\n'.format(time.time()-start_time))
    
    ## TRAIN WITHOUT VAE ##
    start_time = time.time()
    cur_time = time.time()
    for i_episode in range(num_eps_without_VAE):
        reward = training_per_eps_without_vae(ac_opt, ac_model, T, discounted_rate)
        # Log result every log_interval episodes
        if i_episode % log_interval == 0:
            print('Episode {}: cumulative reward: {}; Time elapsed: {} seconds\n'\
                  .format(i_episode, reward, time.time()-cur_time))
            rewards.append(reward)
            cur_time = time.time()
    print('\nTraining completed. Time taken in total: {} seconds\n'.format(time.time()-start_time))
    
    return rewards