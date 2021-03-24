import numpy as np
import torch
from torch import optim
from torch.distributions import Categorical, multivariate_normal
from torch.nn import functional as F
from net_model_3 import VAE, LatentRL
from collections import namedtuple
from utils import initialize, step, updateX, updateZ, decode_pic, get_batch_sample
import time
import matplotlib.pyplot as plt

# Variables get updated every timestep: X, area, peri
# X is refreshed every initialize_interval episodes

# Helper types
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Fixed hyperparams:
beta = 0.0001        
eps = 0.01 # Helper when normalize cumulative rewards
lambda1 = 0.1 # artificial reward vs actor-critic loss
lambda2 = 0.2 # helper hyperparam so that binary cross entropy can be calculated on 0-1 grid
# GPU to use
#device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
device = torch.device("cpu")
def loss_function(x_recon, x, mu, logvar):
    ''' 
    Reconstruction + KL divergence losses summed over all elements and batch.
    See Kingma et al.
    '''
    BCE = F.binary_cross_entropy(lambda2*x_recon+0.5*(1-lambda2),\
                                 lambda2*x+0.5*(1-lambda2), reduction='mean')
    KLD = beta * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    
    return BCE+KLD

def pretrain_vae(vae_model, vae_opt, num_iter, sample_size=1024):
    num_epoch = 1000; #total_loss = 0
    for i_iter in range(num_iter):
        # Sample a mini-batch and then train by shuffle this minibatch
        X = get_batch_sample(sample_size) # 1024 x 1 x 32 x 32
        # compute total_loss for log purpose
        #total_loss = 0
                
        for i_epoch in range(num_epoch):
            vae_opt.zero_grad()
            # Shuffle mini-batch
            x = X[torch.randperm(sample_size)[:32],:,:,:]
            x = x.to(device)
            # Forward and calculate loss
            mu, logvar = vae_model.encode(x)
            x_recon = vae_model.decode(vae_model.reparameterize(mu, logvar))
            
            loss = loss_function(x_recon, x, mu, logvar)
            diff = F.mse_loss(x_recon, x)
            if i_epoch % 1000 == 0:
                print('MSE loss in {}th epoch of the {}th iteration is {}'\
                      .format(i_epoch+1, i_iter+1, diff.item()))
            # Grad descent
            loss.backward()
            vae_opt.step()
            # Update total_loss
            #total_loss += loss
        x = X[torch.randperm(sample_size)[:32],:,:,:]
        #visualize_grid(vae_model, x, output="./pretrain/image")
    print("Finished Pre Train")

        
def sample(vae_model, ac_model, p, z, X, area, peri):
    ''' 
    Given previous (x, z, action) data
    We sample the next (x, z, action) data and update other values accordingly
    '''
    x = vae_model.x_buf[-1]
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
    # Get how much change v from actions (modeled as a Gaussian)
    m = multivariate_normal.MultivariateNormal(loc=action[16:], covariance_matrix=2*torch.eye(8).to(device))
    v = m.sample()
    log_probs.append(m.log_prob(v))
    
    # Save action (log_prob) and artificial reward
    ac_model.saved_actions.append(SavedAction(torch.stack(log_probs), value))
    ac_model.net_rewards.append(net_reward)
    
    # Transition to next z1 given current z, modifying position of z, and how much to add
    # x is updated so that x can be fed into the next iteration
    z1 = step(z, p, action[16:], device)
    x = vae_model.decode(z1.unsqueeze(0))
    vae_model.x_buf.append(x.squeeze(0).clone().detach())
    
    # Get true reward
    area, peri = updateX(X, area, peri, p, x)
    if peri == 0:
        r = float(0)
    else:
        r = float(np.sqrt(area)/peri)
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

    reward_loss = F.mse_loss(torch.tensor(ac_model.rewards),\
                torch.stack(ac_model.net_rewards).squeeze(), reduction='mean')

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
    
    return_reward = ac_model.rewards[-1]
    del ac_model.rewards[:]
    del ac_model.saved_actions[:]
    del ac_model.net_rewards[:]
    return return_reward, reward_loss
        
def training_per_eps(vae_opt, vae_model, ac_opt, ac_model,\
                     p, z, X, area, peri, T, gamma):
    """ 
    Sampling, then train VAE, then train actor critic for one episode.
    Then refresh.
    ac_model: RL net (actor+policy+reward net)
    X: is the big image updated in small parts overtime
    """

    for t in range(T):
        area, peri = sample(vae_model, ac_model, p, z, X, area, peri)
    vae_loss = trainVAE_per_eps(vae_opt, vae_model)
    reward, reward_loss = trainAC_per_eps(ac_opt, ac_model, gamma)
    return area, peri, vae_loss, reward, reward_loss

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
    ac_model = LatentRL()
    ac_opt = optim.Adam(ac_model.parameters(), lr = learning_rate)
    
    # Training
    print('Begin training...\n')
    start_time = time.time()

    # Pretrain the VAE model   
    print('Pretrain VAE:\n')
    cur_time = start_time
    pretrain_vae(vae_model, vae_opt, num_iter_pretrain_vae)
    torch.save(vae_model.state_dict(), 'models/vae_model1')
    print('\nFinish pretrainning for VAE. This takes {} seconds'.format(time.time()-cur_time))
    
    # Declare data for main training
    # Large image X get a small part updated each time step and refresh every 10 episodes
    X = np.zeros((512, 512))
    rewards = [] # rewards log
    
    ## TRAIN WITH VAE ##
    cur_time = time.time()
    for i_episode in range(num_eps_with_VAE):
        if i_episode % initialize_interval == 0:
            # Initialize X and z every 10 episodes
            area, peri, X = initialize()
            #print(X.sum())
            #visualize_grid(None, [torch.tensor(X)], output="./agent_with_vae/initial")
            init_reward = np.sqrt(area)/peri
            # Set each (8, :, :,) of z to be the encoding of a cell in the subgrid of X
            z = decode_pic(X, vae_model, device)
            p = np.zeros(4).astype(int)
            # Pick the top left corner as the first subgrid to put in the x buffer
            tmp = torch.tensor(X[0:32, 0:32], dtype=torch.float).unsqueeze(0).to(device)
            if len(vae_model.x_buf) == 0:
                vae_model.x_buf.append(tmp)
            else:
                vae_model.x_buf[0] = tmp
        # training one episode
        area, peri, vae_loss, reward, reward_loss = training_per_eps(vae_opt, vae_model, \
                        ac_opt, ac_model, p, z, X, area, peri, T, discounted_rate) 
        # Log result every log_interval episodes
        if i_episode % log_interval == 0:
            # Log VAE loss and current cumulative reward
            print('Episode {}: init_reward:{}, cumulative reward: {}; Time elapsed: {} seconds\n'\
                  .format(i_episode+1, init_reward, reward, time.time()-cur_time))
            rewards.append(reward)
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