import numpy as np
import cv2

import torch

import gym
from gym import spaces
from gym.utils import seeding

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from common.utils import spline_interp, extract_geometry, initialize_single, decode_pic

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class IsoperiEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    # xk, yk: knot coordinates; xg, yg: grid coordinates
    def __init__(self, ae_model, grid_dim=8, subgrid_dim=16, 
                 latent_dim=16, resolution=50):
        super(IsoperiEnv, self).__init__()        
        ## Define space ##
        self.max_val = 4; self.min_val = -4 
        # State is (x, z) where x is the current active subgrid, and z is combined latent rep of X
        # z is used solely for inferring p
        self.observation_space = spaces.Dict({'subgrid': spaces.Box(low=self.min_val, high=self.max_val, shape=(1, subgrid_dim, subgrid_dim), dtype=np.float32),
                    'grid_compressed': spaces.Box(low=self.min_val, high=self.max_val, shape=(latent_dim, grid_dim, grid_dim), dtype=np.float32)})
        # Action is (v, p). Here p is where x lies in X, and v is the amount to change x
        self.action_space = spaces.Dict({'change': spaces.Box(low=self.min_val, high=self.max_val, shape=(1, subgrid_dim, subgrid_dim), dtype=np.float32),
                    'position': spaces.Box(low=0, high=1, shape=(grid_dim**2, ), dtype=np.float32)})
        # state is the level-set function phi
        self.state = None   
        self.num_step = 0
        
        # Define various dims
        self.grid_dim = grid_dim
        self.subgrid_dim = subgrid_dim
        self.latent_dim = latent_dim
        
        # Define autoencoder
        self.ae_model = ae_model
        
        # knot and fix grid
        self.xk, self.yk = np.mgrid[-1:1:subgrid_dim+0j, -1:1:subgrid_dim+0j]
        self.xg, self.yg = np.mgrid[-1:1:resolution+0j, -1:1:resolution+0j]

        # Other useful fields
        self.seed()
        
        # First reward
        self.reward = 1
    
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
        eps = 0.001
        s = self.subgrid_dim
        self.X[i*s:(i+1)*s, j*s:(j+1)*s] += eps*act['change']
        self.state['subgrid'] = self.X[i*s:(i+1)*s, j*s:(j+1)*s]
        # Updating z by calling vae model
        mu, logvar = self.ae_model.encode(self.state['subgrid'].squeeze(1))
        z_small = self.ae_model.reparameterize(mu, logvar)
        self.z[:, i, j] = np.copy(z_small)
        
        # Interpolate main values with bicubic spline for the chosen subgrid (to be changed)
        img = spline_interp(self.xk, self.yk, self.state['subgrid'].reshape(self.xk.shape[0], self.yk.shape[0]), self.xg, self.yg)
        new_area, new_peri = extract_geometry(img)
        
        # Update new area, peri arrays and their total values
        self.total_area += new_area - self.area[i][j]
        self.total_peri += new_peri - self.peri[i][j]
        self.area[i][j], self.peri[i][j] = new_area, new_peri
        
        # Find reward
        reward = np.sqrt(abs(self.total_area))/abs(self.total_peri)
        
        # Stop when num of step is more than 100
        done = False
        if self.num_step >= 100 or (self.num_step >= 8 and reward <= self.reward):
            done = True
    
        if self.num_step == 1:
            self.reward = reward
            
        return np.array(self.state), reward, done, {}

    def reset(self):
        return self.reset_at()
        
    def reset_at(self, shape='random', 
                 X=np.array([]),
                 z=np.array([]),
                 area=np.array([]), peri=np.array([]),
                 total_area=0, total_peri=0):
        self.num_step = 0
        
        if shape=='chosen':
            self.X = X
            self.z = z
            self.area, self.peri = area, peri
            self.total_area, self.total_peri = total_area, total_peri
        
        if shape=='random':
            # Get big grid X
            self.area, self.peri, self.total_area, self.total_peri, self.X = initialize_single(
                grid_dim=8, subgrid_dim=16, mode='mountain', 
                num_milestone=11, spacing=6, start_horizontal=10,
                xg=None, yg=None)
            # Transform X to encoder to get compressed grid z
            self.z = decode_pic(X, self.grid_dim, self.subgrid_dim, 
                                self.latent_dim, self.ae_model, device)
            
        # Define state to be a subgrid of X and grid compressed z
        # Choose first subgrid at initialization to be in the left upper corner
        self.state = dict({'subgrid': self.X[:self.subgrid_dim, :self.subgrid_dim], 
                           'grid_compressed': self.z})
        
        return self.state
    
                             