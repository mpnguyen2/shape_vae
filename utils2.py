""" 
This file consists of helper functions 
 
X is the giant image but each time one small part of it is modified

Types: 
1. np_arrays: X, p, v
2. tensors: x, z, z1
"""
import numpy as np
import torch

def initialize(b):
    """ 
    Initialize a shape (represented by bitmap images of dim 512*512):
    Choose 16 random up milestone and down milestone and interpolate to get 
    the up and down heights of the shape along the horizontal[random+128:random+128+256]
    """
    
    X = np.zeros((b, 512, 512), dtype=int)
    X1 = 0.5*np.random.rand(b, 512, 512) # real-valued of X
    milestoneX_up = np.zeros((b, 17))
    milestoneX_down = np.zeros((b, 17))
    for i in range(17):
        milestoneX_up = np.random.randint(low=32, high=128, size=(b, 17))
        milestoneX_down = np.random.randint(low=32, high=128, size=(b, 17))
    k = 128 + np.random.randint(low=-60, high=60, size=b)
    for i in range(16):
        for j in range(16):
            up = (milestoneX_up[:, i]*(1.0-(j/16.0)) + milestoneX_up[:, i+1]*(j/16.0)).astype(int)
            down = (milestoneX_down[:, i]*(1.0-(j/16.0)) + milestoneX_down[:, i+1]*(j/16.0)).astype(int)
            for b1 in range(b):
                X[b1, k[b1]+j, 256-down[b1]: 256+up[b1]] = np.ones(up[b1]+down[b1])
                X1[b1, k[b1]+j, 256-down[b1]: 256+up[b1]] = 0.5*np.random.rand(up[b1]+down[b1])+0.5
        k = k + 16
    area = np.sum(X, axis=(1, 2))
    peri = np.zeros(b)
    peri = 4*area - 2*(np.sum(np.logical_and(X[:,1:,:], X[:,:-1,:]), axis=(1, 2)) \
                       + np.sum(np.logical_and(X[:,:,1:], X[:,:,:-1]), axis=(1, 2)))
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
           
    return torch.tensor(x, dtype=torch.float)
        
def get_batch_sample(batch_size):
    x = torch.zeros(batch_size, 32, 32, dtype=torch.float)
    for b1 in range(batch_size):
        x[b1,:,:] = get_sample()
    return x
    
def decode_pic(X1, vae_model, device):
    """
    Decode each subgrid of X to a low-dim vector and put together to get z
    Return z as a tensor
    """
    b = X1.shape[0]
    z = torch.tensor(np.zeros((b, 8, 16, 16)), dtype=torch.float)
    for i in range(16):
        for j in range(16):
            mu, logvar = vae_model.encode(torch.tensor(X1[:, i*32:(i+1)*32,\
                        j*32:(j+1)*32], dtype=torch.float).to(device))         
            z[:, :, i, j] = vae_model.reparameterize(mu, logvar).cpu().squeeze()
    return z                  

def subgrid_ind(p):
    """
    Find indices of the changing subgrid given changing quadrants vector p
    """
    cx, cy = np.zeros(p.shape[0]).astype(int), np.zeros(p.shape[0]).astype(int)
    for i in range(4):
        cx = cx*2 + (p[:,i]%2)
        cy = cy*2 + (p[:,i]//2)
    return cx.astype(int), cy.astype(int)

def step(z, p, v, device):
    """
    Simulate the dynamics of latent state and output both the true reward.
    The matrix Q is taken to be Id because we assume z1 and v has the same dim 8
    z: old value of z (dim: (8, 16, 16))
    p: position of the subgrid to be updated
    v: how much to update
    Return tensor z1
    """
    b = z.shape[0]
    eps = 0.2
    v_dat = v.cpu().detach().numpy()
    #if np.random.rand() < 0.2:
        #v_dat = np.random.rand(8)-0.5
    cx, cy = subgrid_ind(p); 
    z1_dat = np.zeros((b, v_dat.shape[1]), dtype=float)
    for b1 in range(b):
        z1_dat[b1,:] = z[b1, :, cx[b1], cy[b1]].cpu().numpy() + eps*v_dat[b1] # z1_dat has shape (b, 8)
    z1_dat = np.minimum(np.maximum(z1_dat, -0.99*np.ones((b, 8))), 0.99*np.ones((b, 8)))
    return torch.tensor(z1_dat, dtype=torch.float).to(device)


def updateX(X, X1, old_area, old_peri, p, x):
    """ Calculate new reward from old reward using only part is modified
    and neighboring cells in X
    X: big image
    r: old reward
    p: position of the subgrid to be updated
    x: the updated new subgrid
    """
    n = int(32); b = X.shape[0]
    # Find position of the changing subgrid
    cx, cy = subgrid_ind(p); 
    # Find np data of the old and the new subgrid
    x_old = np.zeros((b, n, n))
    for b1 in range(b):
        x_old[b1,:,:] = X[b1, cx[b1]*n:(cx[b1]+1)*n, cy[b1]*n:(cy[b1]+1)*n]
    x1_new = x.cpu().clone().detach().numpy()
    x_new = np.array(x1_new > 0.5, dtype=float)
    
    # Calculate area by tracking the difference of number of 1 squares
    area = old_area + np.sum(x_new, axis=(1,2)) - np.sum(x_old, axis=(1,2))
    
    ## Calculate perimeter by tracking the difference btw old and new image##
    # First calculate 2 times num of square (change)
    peri = (old_peri+4*(area-old_area))/2
    # Then minus vertical edge shared by two 1 square (change)
    peri += np.sum(np.logical_and(x_old[:,:,1:],x_old[:,:,:-1]), axis=(1,2))\
        - np.sum(np.logical_and(x_new[:,:,1:],x_new[:,:,:-1]), axis=(1,2))
    # Then minus horizontal edge shared by two 1 square (change)
    peri += np.sum(np.logical_and(x_old[:,1:,:],x_old[:,:-1,:]), axis=(1,2))\
        - np.sum(np.logical_and(x_new[:,1:,:],x_new[:,:-1,:]), axis=(1,2))
    # Corner case in the edges of the subsquare:
    for b1 in range(b):
        if cy[b1] > 0:
            peri[b1] += np.sum(np.logical_and(x_old[b1,:, 0], X[b1,cx[b1]*n:(cx[b1]+1)*n, cy[b1]*n-1])) \
                - np.sum(np.logical_and(x_new[b1,:, 0], X[b1,cx[b1]*n:(cx[b1]+1)*n, cy[b1]*n-1]))
        if cy[b1] < 15:
            peri[b1] += np.sum(np.logical_and(x_old[b1,:, -1], X[b1,cx[b1]*n:(cx[b1]+1)*n, (cy[b1]+1)*n])) \
                - np.sum(np.logical_and(x_new[b1,:, -1], X[b1,cx[b1]*n:(cx[b1]+1)*n, (cy[b1]+1)*n]))
        if cx[b1] > 0:
            peri[b1] += np.sum(np.logical_and(x_old[b1, 0, :], X[b1, cx[b1]*n-1, cy[b1]*n:(cy[b1]+1)*n])) \
                - np.sum(np.logical_and(x_new[b1, 0, :], X[b1, cx[b1]*n-1, cy[b1]*n:(cy[b1]+1)*n]))
        if cx[b1] < 15:
            peri[b1] += np.sum(np.logical_and(x_old[b1, -1, :], X[b1, (cx[b1]+1)*n, cy[b1]*n:(cy[b1]+1)*n])) \
                - np.sum(np.logical_and(x_new[b1, -1, :], X[b1, (cx[b1]+1)*n, cy[b1]*n:(cy[b1]+1)*n]))
    
    # The above process gives us only half the perimeter
    peri *= 2
            
    # Finally update X and X1
    for b1 in range(b):
        X1[b1, cx[b1]*n:(cx[b1]+1)*n, cy[b1]*n:(cy[b1]+1)*n] = x1_new[b1,:,:]
        X[b1, cx[b1]*n:(cx[b1]+1)*n, cy[b1]*n:(cy[b1]+1)*n] = x_new[b1,:,:]

    return area, peri


def updateZ(z, p, z1, device):
    """
    Update the whole latent space z by replacing its position at p by v

    """
    b = z.shape[0]
    z = z.cpu().clone().detach().numpy()
    cx, cy = subgrid_ind(p)
    for b1 in range(b):
        z[b1, :, cx[b1], cy[b1]] = z1[b1, :].cpu().detach().numpy()
    return torch.tensor(z).to(device)
