# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:19:49 2021

@author: nguyn
"""

""" 
This file consists of helper functions 
 
X is the giant image but each time one small part of it is modified

Types: 
1. np_arrays: X, p, v
2. tensors: x, z, z1
"""

import numpy as np
import torch
import cv2 


eps = 0.05
eps_act = 0.01

# Get global basis for space of matrix for easier pertubation implementation
# (that is needed in Lipschitz autoencoder)
B = np.zeros((32, 32, 32, 32))
for i in range(32):
    for j in range(32):
        B[i, j] = np.zeros((32, 32))
        B[i, j, i, j] = 1
# Import circle image      
img = cv2.imread('initial_images/circle.bmp')
img = img[:,:,1]
img = 1-np.round(img/255)

def initialize(device):
    x = np.zeros((32, 32), dtype=int)
    milestone_up = np.zeros(4)
    milestone_down = np.zeros(4)
    milestone_up = np.random.randint(low=1, high=12, size=7)
    milestone_down = np.random.randint(low=1, high=12, size=7)
    
    n = 4 # smoothout constant
    orient = np.random.rand()
    if orient < 0.5:
        k = np.random.randint(low=0, high=7)
        for i in range(6):
            for j in range(n):
                up = int(milestone_up[i]*(1.0-(j/n)) + milestone_up[i+1]*(j/n))
                down = int(milestone_down[i]*(1.0-(j/n)) + milestone_down[i+1]*(j/n))
                x[k+j, 16-down:16+up] = np.ones(up+down)
            k += n
    else:
        k = np.random.randint(low=0, high=7)
        for i in range(6):
            for j in range(n):
                up = int(milestone_up[i]*(1.0-(j/n)) + milestone_up[i+1]*(j/n))
                down = int(milestone_down[i]*(1.0-(j/n)) + milestone_down[i+1]*(j/n))
                x[16-down:16+up, k+j] = np.ones(up+down)
            k += n
   
    area = np.sum(x)
    peri = 4*area - 2*(np.sum(np.logical_and(x[1:,:], x[:-1,:])) \
                       + np.sum(np.logical_and(x[:,1:], x[:,:-1])))
    
    return area, peri, torch.tensor(x, dtype=torch.float).to(device)

def random_circ(k):
    img1 = np.copy(img)
    img11 = np.ones((32,32), dtype=float)
    for _ in range(k):
        i = np.random.randint(low=0, high=32)
        j = np.random.randint(low=0, high=32) 
        img1[i,j] = 1-img1[i,j]
        if img1[i,j]==1:
            img11[i,j] = 0.5 + 0.5*np.random.rand()
        else:
            img11[i,j] = np.random.rand()
    return img1, img11
        
def initializeAE(device, int_entry):
    # Generate sample
    x = np.zeros((32, 32), dtype=int)
    x1 = np.random.rand(32, 32)*0.5
    
    milestone_up = np.zeros(4)
    milestone_down = np.zeros(4)
    milestone_up = np.random.randint(low=1, high=12, size=7)
    milestone_down = np.random.randint(low=1, high=12, size=7)
    
    n = 4 # smoothout constant
    orient = np.random.rand()
    if orient < 0.4:
        k = np.random.randint(low=0, high=7)
        for i in range(6):
            for j in range(n):
                up = int(milestone_up[i]*(1.0-(j/n)) + milestone_up[i+1]*(j/n))
                down = int(milestone_down[i]*(1.0-(j/n)) + milestone_down[i+1]*(j/n))
                x[k+j, 16-down:16+up] = np.ones(up+down)
                if int_entry == False: 
                    x1[k+j, 16-down:16+up] = np.random.rand(up+down)*0.5+0.5
            k += n
    elif orient < 0.8:
        k = np.random.randint(low=0, high=7)
        for i in range(6):
            for j in range(n):
                up = int(milestone_up[i]*(1.0-(j/n)) + milestone_up[i+1]*(j/n))
                down = int(milestone_down[i]*(1.0-(j/n)) + milestone_down[i+1]*(j/n))
                x[16-down:16+up, k+j] = np.ones(up+down)
                if int_entry == False: 
                    x1[16-down:16+up, k+j] = np.random.rand(up+down)*0.5+0.5
            k += n
    else:
        k = np.random.randint(low=0, high=3)*5
        x, x1 = random_circ(k)
    # Compute area and peri of sample
    area = np.sum(x)
    peri = 4*area - 2*(np.sum(np.logical_and(x[1:,:], x[:-1,:])) \
                       + np.sum(np.logical_and(x[:,1:], x[:,:-1])))
    
    if int_entry == False:
        return area, peri, torch.tensor(x1, dtype=torch.float).to(device)
    return area, peri, torch.tensor(x, dtype=torch.float).to(device)

def get_batch_sample(batch_size, device, int_entry=False):
    X = torch.zeros(batch_size, 32, 32, dtype=torch.float).to(device)
    R = torch.zeros(batch_size, dtype=torch.float).to(device)
    for i in range(batch_size):
        area, peri, X[i,:,:] = initializeAE(device, int_entry)
        R[i] = torch.tensor(np.sqrt(area)/peri).to(device)
    return X, R

def get_reward_sample(ae_model, batch_size, device):
    Z = torch.rand(batch_size, 80, dtype=torch.float).to(device)
    X = ae_model.decode(Z)
    X_dat = X.cpu().detach().numpy()
    R = torch.zeros(batch_size, dtype=torch.float).to(device)
    for i in range(batch_size):
        R[i] = torch.tensor(reward(X_dat[i,:,:])).to(device)
    return Z, R

def t(dat, device):
    return torch.tensor(dat).to(device)

def step(z, v, device):
    """
    Transition function
    """
    z_dat = z.cpu().detach().numpy().squeeze()
    v_dat = v.cpu().detach().numpy().squeeze()
    z_dat = eps_act*v_dat + (1-eps_act)*z_dat
    
    return t(z_dat, device).unsqueeze(0)

def reward(x):
    x = x > 0.5
    area = np.sum(x)
    peri = 4*area - 2*(np.sum(np.logical_and(x[1:,:], x[:-1,:])) \
                       + np.sum(np.logical_and(x[:,1:], x[:,:-1])))
    return np.sqrt(area)/peri

def pertube(X):
    X1 = np.zeros(X.shape)
    n = X.shape[0]
    for i in range(n):
        sgn = -1
        if np.random.rand() < 0.5:
            sgn = 1
        X1[i, :, :] = X[i,:,:] + eps*sgn*B[np.random.randint(low=0, high=32), np.random.randint(low=0, high=32)]
        X1[i, :, :] = np.maximum(np.minimum(X1[i, :, :], 1), 0)
    return X1