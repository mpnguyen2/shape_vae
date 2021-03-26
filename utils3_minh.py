""" 
This file consists of helper functions 
 
X is the giant image but each time one small part of it is modified

Types: 
1. np_arrays: X, p, v
2. tensors: x, z, z1
"""

import numpy as np
import torch

def initialize():
    """ 
    Initialize a shape (represented by bitmap images of dim 512*512):
    Choose 16 random up milestone and down milestone and interpolate to get 
    the up and down heights of the shape along the horizontal[random+128:random+128+256]
    """
    X = np.zeros((512, 512), dtype=int)
    X1 = 0.4*np.random.rand(512, 512) # real-valued of X
    milestoneX_up = np.zeros(17)
    milestoneX_down = np.zeros(17)
    for i in range(17):
        milestoneX_up = np.random.randint(low=32, high=128, size=17)
        milestoneX_down = np.random.randint(low=32, high=128, size=17)
    k = 128 + np.random.randint(low=-60, high=60)
    for i in range(16):
        for j in range(16):
            up = int(milestoneX_up[i]*(1.0-(j/16.0)) + milestoneX_up[i+1]*(j/16.0))
            down = int(milestoneX_down[i]*(1.0-(j/16.0)) + milestoneX_down[i+1]*(j/16.0))
            X[k+j, 256-down: 256+up] = np.ones(up+down)
            X1[k+j, 256-down:256+up] = 0.4*np.random.rand(up+down)+0.6
        k += 16
    area = np.sum(X)
    peri = 4*area - 2*(np.sum(np.logical_and(X[1:,:], X[:-1,:])) \
                       + np.sum(np.logical_and(X[:,1:], X[:,:-1])))
    return area, peri, X, X1

def get_sample():
    """
    Get meaningful samples for VAE pretraining

    """
    x = 0.5*np.random.rand(32, 32)
    milestoneX_up = np.zeros(5)
    milestoneX_down = np.zeros(5)
    # random variable showing orientation
    orient = np.random.rand()
    # both top and bottom
    if orient < 0.2:
        for i in range(5):
            milestoneX_up = np.random.randint(low=6, high=12, size=5)
            milestoneX_down = np.random.randint(low=6, high=12, size=5)
        k = 0
        for i in range(4):
            for j in range(8):
                up = int(milestoneX_up[i]*(1.0-(j/8.0)) + milestoneX_up[i+1]*(j/8.0))
                down = int(milestoneX_down[i]*(1.0-(j/8.0)) + milestoneX_down[i+1]*(j/8.0))
                x[k+j, 16-down:16+up] = 0.6+0.4*np.random.rand(up+down)
            k += 8
    # only top, bottom, right or left:
    else:     
        milestone = np.random.randint(low=12, high=24, size=5)
        len_one = np.zeros(32).astype(int)
        for i in range(4):
            for j in range(8):
                len_one[8*i+j]= int(milestone[i]*(1.0-(j/8.0)) + milestone[i+1]*(j/8.0))
        # only top
        if orient < 0.4:
            for i in range(32):
                x[i,:len_one[i]] = 0.7+0.3*np.random.rand(len_one[i])
        # only bottom
        elif orient < 0.6:
            for i in range(32):
                x[i, 32-len_one[i]:32] = 0.7+0.3*np.random.rand(len_one[i])
        # only left
        elif orient < 0.8:
            for i in range(32):
                x[:len_one[i], i] = 0.7+0.3*np.random.rand(len_one[i])
        # only right
        else:
            for i in range(32):
                x[32-len_one[i]:32, i] = 0.7+0.3*np.random.rand(len_one[i])
           
    x = x.reshape(1, 32, 32)
    return torch.tensor(x, dtype=torch.float)
        
def get_batch_sample(batch_size):
    x = torch.zeros(batch_size, 1, 32, 32, dtype=torch.float)
    for i in range(batch_size):
        x[i,:,:,:] = get_sample()
    return x
    
def decode_pic(X1, vae_model, device):
    """
    Decode each subgrid of X to a low-dim vector and put together to get z
    Return z as a tensor
    """
    z = torch.tensor(np.zeros((8, 16, 16)), dtype=torch.float).to(device)
    for i in range(16):
        for j in range(16):
            mu, logvar = vae_model.encode(torch.tensor(X1[i*32:(i+1)*32,\
                        j*32:(j+1)*32], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device))         
            z[:, i, j] = vae_model.reparameterize(mu, logvar).squeeze()
    return z                  

def subgrid_ind(p):
    """
    Find indices of the changing subgrid given changing quadrants vector p
    """
    cx, cy = 0, 0
    for i in range(4):
        p[i] = int(p[i])
        cx = cx*2 + (p[i]%2)
        cy = cy*2 + (p[i]//2)
    return int(cx), int(cy)

'''
def step(z, p, v, device):
    """
    Simulate the dynamics of latent state and output both the true reward.
    The matrix Q is taken to be Id because we assume z1 and v has the same dim 8
    z: old value of z (dim: (16, 16, 8))
    p: position of the subgrid to be updated
    v: how much to update
    Return tensor z1
    """
    eps = 0.6
    v_dat = v.detach().numpy()
    if np.max(np.abs(v_dat)) != 0:
        v_dat = v_dat/np.max(np.abs(v_dat))
        
    cx, cy = subgrid_ind(p)
    z1_dat = z[:, cx, cy].detach().numpy() + eps*v_dat
    z1_dat = np.minimum(z1_dat, 0.99)
    return torch.tensor(z1_dat).to(device)
'''

def updateX(X, X1, old_area, old_peri, p, x):
    """ Calculate new reward from old reward using only part is modified
    and neighboring cells in X
    X: big image
    r: old reward
    p: position of the subgrid to be updated
    x: the updated new subgrid
    """
    n = 32
    # Find position of the changing subgrid
    cx, cy = subgrid_ind(p)
    # Find np data of the old and the new subgrid
    x_old = X[cx*n:(cx+1)*n, cy*n:(cy+1)*n]
    x1_new = x.cpu().squeeze().clone().detach().numpy()
    x_new = np.array(x1_new > 0.5, dtype=int)
    
    # Calculate area by tracking the difference of number of 1 squares
    area = old_area + np.sum(x_new) - np.sum(x_old)
    
    ## Calculate perimeter by tracking the difference btw old and new image##
    # First calculate 2 times num of square (change)
    peri = (old_peri+4*(area-old_area))/2
    # Then minus vertical edge shared by two 1 square (change)
    peri += np.sum(np.logical_and(x_old[:,1:],x_old[:,:-1]))\
        - np.sum(np.logical_and(x_new[:,1:],x_new[:,:-1]))
    if cy > 0:
        peri += np.sum(np.logical_and(x_old[:, 0], X[cx*n:(cx+1)*n, cy*n-1])) \
            - np.sum(np.logical_and(x_new[:, 0], X[cx*n:(cx+1)*n, cy*n-1]))
    if cy < 15:
        peri += np.sum(np.logical_and(x_old[:, -1], X[cx*n:(cx+1)*n, (cy+1)*n])) \
            - np.sum(np.logical_and(x_new[:, -1], X[cx*n:(cx+1)*n, (cy+1)*n]))
    # Then minus horizontal edge shared by two 1 square (change)
    peri += np.sum(np.logical_and(x_old[1:,:],x_old[:-1,:]))\
        - np.sum(np.logical_and(x_new[1:,:],x_new[:-1,:]))
    if cx > 0:
        peri += np.sum(np.logical_and(x_old[0, :], X[cx*n-1, cy*n:(cy+1)*n])) \
            - np.sum(np.logical_and(x_new[0, :], X[cx*n-1, cy*n:(cy+1)*n]))
    if cx < 15:
        peri += np.sum(np.logical_and(x_old[-1, :], X[(cx+1)*n, cy*n:(cy+1)*n])) \
            - np.sum(np.logical_and(x_new[-1, :], X[(cx+1)*n, cy*n:(cy+1)*n]))
    
    # The above process gives us only half the perimeter
    peri *= 2
            
    # Finally update X and X1
    X[cx*n:(cx+1)*n, cy*n:(cy+1)*n] = x_new
    X1[cx*n:(cx+1)*n, cy*n:(cy+1)*n] = x1_new

    return area, peri


def updateZ(z, p, z1, device):
    """
    Update the whole latent space z by replacing its position at p by v

    """
    z = z.cpu().clone().detach().numpy()
    cx, cy = subgrid_ind(p)
    z[:, cx, cy] = z1.cpu().detach().numpy()
    return torch.tensor(z).to(device)

def updateX_no_reward(X, X1, p, x):
    n = 32
    cx, cy = subgrid_ind(p)
    x1_new = x.cpu().squeeze().clone().detach().numpy()
    x_new = np.array(x1_new > 0.5, dtype=int)
    X[cx*n:(cx+1)*n, cy*n:(cy+1)*n] = x_new
    X1[cx*n:(cx+1)*n, cy*n:(cy+1)*n] = x1_new