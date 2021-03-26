# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:34:39 2021

@author: nguyn
"""

"""
Testing files.
"""
import torch
import numpy as np
import cv2
from net_model_3_minh import VAE, LatentRL
from train3_minh import train
from utils3_minh import initialize, decode_pic, updateZ, updateX_no_reward

#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

def run_traj(vae_model_path, ac_model_path, out_file, num_step):
    # Load models and initialize variables
    vae_model = VAE().to(device); ac_model = LatentRL().to(device)
    vae_model.load_state_dict(torch.load(vae_model_path))
    ac_model.load_state_dict(torch.load(ac_model_path))
    _, _, X, X1 = initialize()
    z = decode_pic(X1, vae_model, device)
    p = np.zeros(4); #v = torch.zeros(8)
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_file, fourcc, 20.0, (512, 512), isColor=False)
    
    # Doing forward pass for num_step steps using loaded models
    for i in range(num_step):
        ## Visualization by adding X to a video ##
        out.write(X)
        #cv2.imshow('frame', X1)
        ## Perform action and update vars after each time step  
        # Perform action -> (p, v)
        action, z1, _ , _ = ac_model(z.unsqueeze(0))
        action = action.squeeze(0)
        ''' DEBUG
        if i %20 == 0:
            print(action)
            print(X.sum())
        '''
        #v = action[16:]
        for i in range(4):
            p[i] = torch.argmax(action[4*i:4*i+4]).item()
        #v = action[16:]
        # Update z, z1, x, X
        z = updateZ(z, p, z1.squeeze(), device)
        x = vae_model.decode(z1)
        updateX_no_reward(X, X1, p, x)
    # Release test video
    out.release()
    #cv2.destroyAllWindows()
    
#### TESTING TRAIN ####

train(num_iter_pretrain_vae=5, num_eps_with_VAE=1000, \
          initialize_interval=10, T=64, discounted_rate=0.99, \
              log_interval=20, save_interval=100)

vae_model_path = 'models/vae_model1'
ac_model_path = 'models/ac_model1'
out_file = 'test_vid/test2.wmv'
num_step = 400
run_traj(vae_model_path, ac_model_path, out_file, num_step)