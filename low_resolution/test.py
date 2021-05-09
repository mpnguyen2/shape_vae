# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:34:39 2021

@author: nguyn
"""

"""
Testing files.
"""
import torch
from torch.distributions import multivariate_normal
import numpy as np
import cv2
from net_model import VAE, LatentRL
from train import train
from utils import initialize, step

#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

def run_traj(vae_model_path, ac_model_path, out_file, num_step, write_interval=20):
    # Load models and initialize variables
    vae_model = VAE().to(device); ac_model = LatentRL().to(device)
    vae_model.load_state_dict(torch.load(vae_model_path))
    ac_model.load_state_dict(torch.load(ac_model_path))
    _, _, x = initialize(device)
    mu, logvar = vae_model.encode(x.unsqueeze(0))
    z = vae_model.reparameterize(mu, logvar).squeeze(0)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_file, fourcc, 20.0, (32, 32), isColor=False)
    
    # Doing forward pass for num_step steps using loaded models
    for i in range(num_step):
        if i % write_interval == 0:
            ## Visualization by adding X to a video ##
            out.write(np.array((1-x.cpu().detach().numpy())*255, dtype=np.uint8))
            #cv2.imshow('frame', X1)
        ## Select action v  
        act_mean, _, _ = ac_model(z.unsqueeze(0))
        #act_mean, act_var, value = act_mean.squeeze(0), \
        #torch.diag(torch.exp(act_logvar.squeeze(0))), value.squeeze(0)
        #m = multivariate_normal.MultivariateNormal(loc=act_mean, covariance_matrix=act_var)
        v = act_mean.squeeze(0)
        
        # Step into next z and update x accordingly
        z = step(z, v, device)
        x = vae_model.decode(z.unsqueeze(0)).squeeze(0)
        x = torch.round(x)
        
    # Release test video
    out.release()
    #cv2.destroyAllWindows()
    
#### TESTING TRAIN ####
'''
train(num_iter_pretrain_vae=10, num_eps_with_VAE=1000, \
          initialize_interval=10, T=128, discounted_rate=0.99, \
              log_interval=20, save_interval=100)
'''
vae_model_path = 'models/vae_model1'
ac_model_path = 'models/ac_model1'
out_file = 'test_vid/test.wmv'
num_step = 10000
run_traj(vae_model_path, ac_model_path, out_file, num_step)
