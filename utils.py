""" 
This file consists of helper functions 
 
X is the giant image but each time one small part of it is modified

Types: 
1. np_arrays: X, p, v
2. tensors: x, z, z1
"""
import numpy as np
import torch

def initialize(X):
    """ 
    Initialize a shape (represented by bitmap images of dim 512*512):
    Choose 16 random up milestone and down milestone and interpolate to get 
    the up and down heights of the shape along the horizontal[128:128+256]
    """
    X = np.zeros((512, 512))
    milestoneX_up = np.zeros(16)
    milestoneX_down = np.zeros(16)
    for i in range(17):
        milestoneX_up = np.random.randint(low=32, high=128, size=17)
        milestoneX_down = np.random.randint(low=32, high=128, size=17)
    k = 128
    for i in range(16):
        for j in range(16):
            up = int(milestoneX_up[i]*(1.0-(j/16.0)) + milestoneX_up[i+1]*(j/16.0))
            down = int(milestoneX_down[i]*(1.0-(j/16.0)) + milestoneX_down[i+1]*(j/16.0))
            X[k+j, 256-down: 256+up] = np.ones(up+down)
        k += 16
    area = np.sum(X)
    peri = 4*area - 2*(np.sum(np.logical_and(X[1:,:], X[:-1,:])) \
                       + np.sum(np.logical_and(X[:,1:], X[:,:-1])))
    return area, peri
        
def decode_pic(X, vae_model):
    """
    Decode each subgrid of X to a low-dim vector and put together to get z
    Return z as a tensor
    """
    z = np.zeros((8, 16, 16))
    for i in range(16):
        for j in range(16):
            mu, logvar = vae_model.encode(torch.tensor(X[i*32:(i+1)*32,\
                        j*32:(j+1)*32], dtype=torch.float).unsqueeze(0).unsqueeze(0))         
            z[:, i, j] = vae_model.reparameterize(mu, logvar).squeeze().detach().numpy()
    return torch.tensor(z)                       

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

def step(z, p, v):
    """
    Simulate the dynamics of latent state and output both the true reward.
    The matrix Q is taken to be Id because we assume z1 and v has the same dim 8
    z: old value of z (dim: (16, 16, 8))
    p: position of the subgrid to be updated
    v: how much to update
    Return tensor z1
    """
    eps = 0.01
    z = z.detach().numpy()    
    cx, cy = subgrid_ind(p)
    return torch.tensor(z[:, cx, cy] + eps*v.clone().detach().numpy())


def updateX(X, old_area, old_peri, p, x):
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
    x_new = x.squeeze().detach().clone().numpy()
    x_new = np.array(x_new > 0.5, dtype=float)
    
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
            
    # Finally update X
    X[cx*n:(cx+1)*n, cy*n:(cy+1)*n] = x_new
    return area, peri


def updateZ(z, p, z1):
    """
    Update the whole latent space z by replacing its position at p by v

    """
    z = z.detach().numpy()
    cx, cy = subgrid_ind(p)
    z[:, cx, cy] = z1.clone().detach().numpy()
    return torch.tensor(z, dtype=torch.float)
