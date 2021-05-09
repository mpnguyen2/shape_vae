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

'''
y = np.zeros((32, 32), dtype=int)
milestone_up = np.zeros(4)
milestone_down = np.zeros(4)
milestone_up = np.random.randint(low=2, high=12, size=7)
milestone_down = np.random.randint(low=2, high=12, size=7)
n = 4 # smoothout constant
k = np.random.randint(low=0, high=7)
for i in range(6):
    for j in range(n):
        up = int(milestone_up[i]*(1.0-(j/n)) + milestone_up[i+1]*(j/n))
        down = int(milestone_down[i]*(1.0-(j/n)) + milestone_down[i+1]*(j/n))
        y[k+j, 16-down:16+up] = np.ones(up+down)
    k += n
area_y = np.sum(y)
peri_y = 4*area_y - 2*(np.sum(np.logical_and(y[1:,:], y[:-1,:])) \
                       + np.sum(np.logical_and(y[:,1:], y[:,:-1])))
'''

y = cv2.imread('initial_images/heart.bmp')
y = y[:,:,1]
y = np.round(y/255)
    
def initialize(device):
    """
    x = np.zeros((32, 32), dtype=int)
    milestone_up = np.zeros(4)
    milestone_down = np.zeros(4)
    milestone_up = np.random.randint(low=2, high=12, size=7)
    milestone_down = np.random.randint(low=2, high=12, size=7)
    
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
    """
    area = np.sum(y)
    peri = 4*area - 2*(np.sum(np.logical_and(y[1:,:], y[:-1,:])) \
                       + np.sum(np.logical_and(y[:,1:], y[:,:-1])))
        
    return area, peri, torch.tensor(y, dtype=torch.float).to(device)
        
def initializeVAE(device):
    x = np.zeros((32, 32), dtype=int)
    milestone_up = np.zeros(4)
    milestone_down = np.zeros(4)
    milestone_up = np.random.randint(low=2, high=12, size=7)
    milestone_down = np.random.randint(low=2, high=12, size=7)
    
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

def get_batch_sample(batch_size, device):
    x = torch.zeros(batch_size, 32, 32, dtype=torch.float).to(device)
    for i in range(batch_size):
        _, _, x[i,:,:] = initializeVAE(device)
    return x

def step(z, v, device):
    """

    """
    eps = 0.001
    v_dat = v.cpu().detach().numpy()
    #if np.linalg.norm(v_dat) != 0:
    #    v_dat = v_dat/np.linalg.norm(v_dat)
    z_dat = z.cpu().detach().numpy()
    norm_z = np.linalg.norm(z_dat)
    z = z_dat + eps*norm_z*v_dat
    
    return torch.tensor(z).to(device)

def reward(x):
    area = np.sum(x)
    peri = 4*area - 2*(np.sum(np.logical_and(x[1:,:], x[:-1,:])) \
                       + np.sum(np.logical_and(x[:,1:], x[:,:-1])))
    return np.sqrt(area)/peri