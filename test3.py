"""
Testing files.
"""
import torch
import numpy as np
import cv2
from net_model_3 import VAE, LatentRL
from train3 import train
from utils import initialize, decode_pic, step, updateZ, updateX_no_reward

device = torch.device("cpu")

#### TESTING TRAIN ####
'''
train(num_iter_pretrain_vae=3, num_eps_with_VAE=2000, \
          initialize_interval=10, T=64, discounted_rate=0.99, \
              log_interval=20, save_interval=100)
''' 

def run_traj(vae_model_path, ac_model_path, out_file, num_step):
    # Load models and initialize variables
    vae_model = VAE().to(device); ac_model = LatentRL()
    vae_model.load_state_dict(torch.load(vae_model_path))
    ac_model.load_state_dict(torch.load(ac_model_path))
    _, _, X = initialize()
    z = decode_pic(X, vae_model, device)
    p = np.zeros(4); v = torch.zeros(8)
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    out = cv2.VideoWriter(out_file, fourcc, 20.0, (512, 512), isColor=False)
    
    # Doing forward pass for num_step steps using loaded models
    for i in range(num_step):
        ## Visualization by adding X to a video ##
        X1 = (255*(X <= 0.5)).astype(np.uint8)
        out.write(X1)
        #cv2.imshow('frame', X1)
        ## Perform action and update vars after each time step  
        # Perform action -> (p, v)
        action, _ , _ = ac_model(z.unsqueeze(0))
        action = action.squeeze(0)
        v = action[16:]
        for i in range(4):
            p[i] = torch.argmax(action[4*i:4*i+4]).item()
        v = action[16:]
        # Update z, z1, x, X
        z1 = step(z, p, v, device)
        z = updateZ(z, p, z1)
        x = vae_model.decode(z1.unsqueeze(0))
        updateX_no_reward(X, p, x)
    # Release test video
    out.release()
    #cv2.destroyAllWindows()