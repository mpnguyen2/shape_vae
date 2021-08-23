# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:34:39 2021

@author: nguyn
"""

"""
Testing files.
"""
import torch
#from torch.distributions import multivariate_normal
import numpy as np
import cv2
from net_model import AE, LatentRL
from train import train
from utils import initialize, step

#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

def t(dat):
    return torch.tensor(dat).to(device)

def run_traj(ae_model_path, rl_model_path, out_file, num_step, write_interval=1):
    # Load models and initialize variables
    rl_model = LatentRL().to(device); 
    rl_model.load_state_dict(torch.load(rl_model_path))
    ae_model = AE().to(device)
    ae_model.load_state_dict(torch.load(ae_model_path))
    # Initialize x and z
    _, _, x = initialize(device)
    z = ae_model.encode(x.unsqueeze(0))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_file, fourcc, 20.0, (32, 32), isColor=False)
    
    # Doing forward pass for num_step steps using loaded models
    for i in range(num_step):
        if i % write_interval == 0:
            ## Visualization by adding X to a video ##
            out.write(np.array((1-x.cpu().detach().numpy())*255, dtype=np.uint8))
            #cv2.imshow('frame', X1)
        act = rl_model.actor(z)
        z = step(z, act, device)
        x = ae_model.decode(z).squeeze()
        
    # Release test video
    out.release()
    #cv2.destroyAllWindows()
    
#### TESTING TRAIN ####

train(newtrain_ae=True, newtrain_reward=False, num_eps=1, 
      num_reward_train=0, num_pretrain_ae=20, 
      initialize_interval=1, T=200, discounted_rate=0.99,
              log_interval=10, save_interval=5)
'''
ae_model_path = 'models/ae_model1'
rl_model_path = 'models/rl_model1'
out_file = 'test_vid/test.wmv'
num_step = 500
run_traj(ae_model_path, rl_model_path, out_file, num_step)
'''