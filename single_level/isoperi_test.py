import numpy as np
import cv2

import torch

import gym
from gym import spaces
from gym.utils import seeding

from isoperi import IsoperiEnv

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from common.utils import spline_interp, extract_geometry, initialize_single, decode_pic

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class IsoperiTestEnv(IsoperiEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, ae_model, grid_dim=8, subgrid_dim=16, 
                 latent_dim=16, resolution=50, 
                 out_file='videos/test2.wmv'):
        super(IsoperiTestEnv, self).__init__(ae_model, 
                                grid_dim, subgrid_dim, 
                                latent_dim, resolution)        
        # Create video writer
        self.out_file = out_file
        fourcc = cv2.VideoWriter_fourcc(*'WMV1')
        self.out = cv2.VideoWriter(self.out_file, fourcc, 20.0, (self.xg.shape[0], self.yg.shape[1]), isColor=False)
        # Other useful fields
        self.seed()
        self.reward = 0
    
    # For probabilistic purpose (unique seed for the obj). To be used.
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        self.num_step += 1
        # Choose position to change the subgrid
        p = np.argmax(act['position'])
        i, j = p//self.grid_dim, p % self.grid_dim
        # Change the subgrid by updating both big grid X and state
        eps = 0.0001
        s = self.subgrid_dim
        self.X[i*s:(i+1)*s, j*s:(j+1)*s] += eps*act['change']
        self.state['subgrid'] = self.X[i*s:(i+1)*s, j*s:(j+1)*s]
        # Updating z by calling vae model
        mu, logvar = self.ae_model.encode(self.state['subgrid'].squeeze(1))
        z_small = self.ae_model.reparameterize(mu, logvar)
        self.z[:, i, j] = np.copy(z_small)
        
        # Interpolate main values with bicubic spline for the chosen subgrid (to be changed)
        img = spline_interp(self.xk, self.yk, self.state['subgrid'].reshape(self.xk.shape[0], self.yk.shape[0]), self.xg, self.yg)
        new_area, new_peri, contours = extract_geometry(img, return_contour=True)
        
        # Update new area, peri arrays and their total values
        self.total_area += new_area - self.area[i][j]
        self.total_peri += new_peri - self.peri[i][j]
        self.area[i][j], self.peri[i][j] = new_area, new_peri
        
        # Find reward
        reward = np.sqrt(abs(self.total_area))/abs(self.total_peri)
        
        # Print info
        if self.num_step % 10 == 1:
            print('Reward: ', reward, '; Step: ', self.num_step)
        # Update frame for trajectory's video rendering
        if self.num_step % 10 == 1:
            img = spline_interp(self.xk, self.yk, self.state['subgrid'].reshape(self.xk.shape[0], self.yk.shape[0]), self.xg, self.yg)
            cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0,255,0), thickness=1)
            self.out.write(img)
            
        # Stop when num of step is more than 2000
        done = False
        if self.num_step >= 4000:
            done = True
            # Save video
            self.out.release()
            
        return np.array(self.state), reward, done, {}

    def reset(self):
        return self.reset_at()
        
    def reset_at(self, shape='random'):
        super().reset_at(shape)

        return self.state