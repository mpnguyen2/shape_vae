# TO BE UPDATED: Training the static version of networks

import numpy as np
import torch
from torch import optim
#from torch.distributions import multivariate_normal
#from torch.distributions.bernoulli import Bernoulli
from torch.nn import functional as F
#from collections import namedtuple
import time

from net_model_static import Reward, AE, LatentRL
from utils import initialize, step, get_batch_sample, get_reward_sample, pertube


# Variables get updated every timestep: X, area, peri
# X is refreshed every initialize_interval episodes

# Helper types
#SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Fixed hyperparams:     
eps = 0.01 # Helper when normalize cumulative rewards
beta1 = 20

# model file names:
reward_model_path = 'models/reward_model1'
ae_model_path = 'models/ae_model1'
rl_model_path = 'models/rl_model1'

# GPU to use
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#device = torch.device("cpu")

# BCE reconstruction loss for autoencoder
def loss_function(x_recon, x):
    BCE = F.binary_cross_entropy(x_recon, x, reduction="mean")
    return BCE

# Pretrain a Lipschitz autoencoder
def pretrain_ae(ae_model, ae_opt, reward_model, num_iter, sample_size=512):
    num_epoch = 3001; 
    for i_iter in range(num_iter):
        # Sample a mini-batch and then train by shuffle this minibatch
        X, _ = get_batch_sample(sample_size, device, int_entry=True) # 1024 x 32 x 32
        # compute total_loss for log purpose
        #total_loss = 0

        for i_epoch in range(num_epoch):
            ae_opt.zero_grad()
            # Shuffle mini-batch
            perm = torch.randperm(sample_size)[:32]      
            x = X[perm,:,:]
            X_p = pertube(X[perm,:,:].cpu().detach().numpy())
            x_p = torch.tensor(X_p, dtype=torch.float).to(device)
            
            # Forward and calculate loss
            x_recon, z = ae_model(x)
            x_recon_p, z_p = ae_model(x_p)
            
            loss = loss_function(x_recon, x) 
            loss += loss_function(x_recon_p, x_p)
            loss += 5*F.smooth_l1_loss(z, z_p, reduction='sum', beta=0.01)
            
            #diff = F.mse_loss(x_recon, x)
            if i_epoch % 1000 == 0:
                print('AE loss in {}th epoch of the {}th iteration is {:.5f}'\
                      .format(i_epoch+1, i_iter+1, loss.item()))
            # Grad descent
            loss.backward()
            ae_opt.step()
            
    print("Finished AE pretrain")

# Pretrain a reward network that takes latent variable z as input and output corresponding reward
def pretrain_reward(ae_model, reward_model, reward_opt, num_iter, sample_size=512):
    num_epoch = 4001; #total_loss = 0
    Z, R = 0, 0 # X, R should have size sample_size x 32 x 32 and 1 x 32 x 32
    for i_iter in range(num_iter):
        # Sample a mini-batch and then train by shuffle this minibatch
        Z, R = get_reward_sample(ae_model, sample_size, device)
        # compute total_loss for log purpose
        #total_loss = 0
                
        for i_epoch in range(num_epoch):
            reward_opt.zero_grad()
            
            # Shuffle mini-batch
            perm = torch.randperm(sample_size)[:32]
            z = Z[perm,:]
            r =R[perm]
            # Forward and calculate loss
            loss = F.smooth_l1_loss((reward_model(z).squeeze()-0.1)*10, 
                            (r-0.1)*10, reduction='sum', beta=0.01)
           
            if i_epoch % 1000 == 0:
                print('Reward loss in {}th epoch of the {}th iteration is {:.5f}'\
                      .format(i_epoch+1, i_iter+1, loss.item()))
            # Grad descent
            loss.backward()
            reward_opt.step()
            
    print("Finished Reward pretrain")

# One step data sample for RL training
def sample(rl_model, reward_model, z):
    # Sample action from policy distribution in current rl_model
    # Record log_prob (already handled in forward fct)
    act = rl_model(z)
    z = step(z, act, device)
    # Take reward: reward of the decoding x of new z
    rl_model.rewards.append(reward_model(z).item())   
    
# Vanilla actor-critic
def trainRL_per_eps(rl_opt, rl_model, gamma):
    """
    Currently just do a simple REINFORCE
    """
    policy_losses = [] #list to save actor loss
    #value_losses = [] #list to save critic loss
    
    # Calculate cumulative reward R backwardly
    cum_rewards = []
    R = 0
    for r in rl_model.rewards[:]:
        R = float(r + gamma*R)
        cum_rewards.insert(0, R)
    
    # Tensorize and normalize cumulative rewards
    cum_rewards = torch.tensor(cum_rewards).to(device)
    cum_rewards = (cum_rewards-cum_rewards.mean())/(cum_rewards.std() + eps)
    
    # Calculate individual terms in policy and critic loss fcts
    for log_prob, value, R in zip(rl_model.log_prob, rl_model.values, cum_rewards):
        advantage = R - value.item()
        # calculate policy losses
        # assume that action only random around the choice of quadrant
        policy_losses.append(-log_prob*advantage)
        # calculate critic losses
        #value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))
    
    rl_opt.zero_grad()
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
    rl_opt.step()
    
    last_reward = rl_model.rewards[-1]
    del rl_model.rewards[:]
    del rl_model.values[:]
    del rl_model.log_prob[:]
    
    return last_reward
        
def training_per_eps(z, reward_model, rl_opt, rl_model, T, gamma):
    """ 
    Sampling, then train VAE, then train actor critic for one episode.
    Then refresh.
    ac_model: RL net (actor+policy+reward net)
    X: is the big image updated in small parts overtime
    """

    for t in range(T):
        sample(rl_model, reward_model, z)
    
    last_reward = trainRL_per_eps(rl_opt, rl_model, gamma)
    return last_reward
    
def train(newtrain_ae, newtrain_reward, num_eps, num_reward_train, num_pretrain_ae, initialize_interval, T, discounted_rate, 
              log_interval, save_interval):
    """
    Training: Train policy network using REINFORCE
    """
    # Build models and corresponding optimizers
    reward_model = Reward().to(device)
    reward_opt = optim.Adam(reward_model.parameters(), lr = 0.003)
    ae_model = AE().to(device)
    ae_opt = optim.Adam(ae_model.parameters(), lr = 0.001)
    rl_model = LatentRL().to(device)
    rl_opt = optim.Adam(rl_model.parameters(), lr = 0.001)
    
    # Training
    start_time = time.time()
    print('Begin pretraining...\n')
    # Pretrain the VAE model 
    if newtrain_ae == True:       
        cur_time = start_time
        print('Pretrain AE:\n')
        pretrain_ae(ae_model, ae_opt, reward_model, num_pretrain_ae)
        torch.save(ae_model.state_dict(), ae_model_path)
        print('\nFinish pretrainning for AE. This takes {:.3f} seconds'.format(time.time()-cur_time))
    else:
        print('Begin loading pretrained ae model\n')
        ae_model.load_state_dict(torch.load(ae_model_path))
        print('Done loading pretrained ae model\n')
        
    # Pretrain reward model
    if newtrain_reward == True:
        cur_time = time.time()
        print('Pretrain reward network:\n') 
        pretrain_reward(ae_model, reward_model, reward_opt, num_reward_train)
        torch.save(reward_model.state_dict(), reward_model_path)
        print('\nFinish pretrainning for reward net. This takes {:.3f} seconds'.format(time.time()-cur_time))  
    else:
        print('Begin loading pretrained reward model\n')
        reward_model.load_state_dict(torch.load(reward_model_path))
        print('Done loading pretrained reward model\n')
    
    # Training
    print('\n\nBegin main training...\n')
    start_time = time.time()
    # Initialize x
    x = torch.zeros(32, 32)  
    rewards = [] # rewards log
    
    ## TRAINING ##
    cur_time = time.time()
    for i_episode in range(num_eps):
        if i_episode % initialize_interval == 0:
            # Initialize X and z every 10 episodes
            area, peri, x = initialize(device)
            init_reward = np.sqrt(area)/peri
            _, z = ae_model(x.unsqueeze(0).to(device))
            
        # training one episode
        reward = training_per_eps(z, reward_model, rl_opt, rl_model, T, discounted_rate) 
        rewards.append(reward)
        # Log result every log_interval episodes
        if i_episode % log_interval == 0:
            # Log VAE loss and current cumulative reward
            print('Episode {}: init_reward:{:.3f}, cumulative reward: {:.3f}; percentage: {:.2f}; Time elapsed: {:.3f} seconds\n'\
                  .format(i_episode+1, init_reward, reward, 100*reward/init_reward, time.time()-cur_time))
            cur_time = time.time()
        if i_episode % save_interval == 0:
            torch.save(rl_model.state_dict(), rl_model_path)
            
    print('\nTraining completed. Time taken: {} seconds\n'.format(time.time()-start_time))
    
    return rewards