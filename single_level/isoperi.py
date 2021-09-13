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
                 latent_dim=16, resolution=50, shape_init='random'):
        super(IsoperiEnv, self).__init__()        
        ## Define space ##
        self.max_val = 4; self.min_val = -4 
        # State is (x, z) where x is the current active subgrid, and z is combined latent rep of X
        # z is used solely for inferring p
        self.change_dim = subgrid_dim**2
        self.pos_dim = (grid_dim**2)*latent_dim
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, 
                                shape=(self.change_dim+self.pos_dim,), dtype=np.float32)
        # Action is (v, p). Here p is where x lies in X, and v is the amount to change x
        self.action_space = spaces.Box(low=self.min_val, high=self.max_val, 
                                shape=(self.change_dim + grid_dim**2, ), dtype=np.float32)
        # state is the level-set function phi
        self.state = np.zeros(self.change_dim+self.pos_dim) 
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
        
        # Same initial shape
        self.shape_init = shape_init
        self.area_init, self.peri_init, self.total_area_init, self.total_peri_init, self.X_init = initialize_single(
                        grid_dim=8, subgrid_dim=16, mode='mountain', 
                        num_milestone=6, spacing=10, start_horizontal=20,
                        xg=None, yg=None)
        
        self.z_init = decode_pic(self.X_init + 0.5, 
                    grid_dim=8, subgrid_dim=16, latent_dim=16, 
                    vae_model=self.ae_model, device=device)
        
        print('Reward start: ', np.sqrt(abs(self.total_area_init))/abs(self.total_peri_init))
            
    # For probabilistic purpose (unique seed for the obj). To be used.
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        self.num_step += 1
        # Choose position to change the subgrid
        p = np.argmax(act[self.change_dim:])
        i, j = p//self.grid_dim, p % self.grid_dim
        
        # Change the subgrid by updating both big grid X and state
        eps = 0.001
        s = self.subgrid_dim
        self.X[i*s:(i+1)*s, j*s:(j+1)*s] += eps*act[:self.change_dim].reshape(s, s)
        self.state[:self.change_dim] = self.X[i*s:(i+1)*s, j*s:(j+1)*s].reshape(-1)
        
        # Updating z by calling vae model
        new_part = self.state[:self.change_dim].reshape(self.subgrid_dim, self.subgrid_dim)+0.5
        mu, logvar = self.ae_model.encode(
                torch.tensor(new_part, dtype=torch.float32).unsqueeze(0).to(device))
        z_small = self.ae_model.reparameterize(mu, logvar).cpu().detach().numpy()
        s = self.grid_dim
        start = (i*s+j+self.change_dim); end = start + (s*s*self.latent_dim); step=s**2
        self.state[start:end:step] = z_small.reshape(-1)
        
        # Interpolate main values with bicubic spline for the chosen subgrid (to be changed)
        img = spline_interp(self.xk, self.yk, new_part.reshape(self.xk.shape[0], self.yk.shape[0]), self.xg, self.yg)
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
        return self.reset_at(self.shape_init)
        
    def reset_at(self, shape='random', 
                 X=np.array([]),
                 z=np.array([]),
                 area=np.array([]), peri=np.array([]),
                 total_area=0, total_peri=0):
        self.num_step = 0
        
        if shape=='chosen':
            self.X = self.X_init
            self.state[self.change_dim:] = self.z_init.reshape(-1)
            self.area, self.peri = self.area_init, self.peri_init
            self.total_area, self.total_peri = self.total_area_init, self.total_peri_init
        
        if shape=='random':
            # Get big grid X
            self.area, self.peri, self.total_area, self.total_peri, self.X = initialize_single(
                grid_dim=8, subgrid_dim=16, mode='mountain', 
                num_milestone=11, spacing=6, start_horizontal=10,
                xg=None, yg=None)
            # Transform X to encoder to get compressed grid z
            self.state[self.change_dim:] = decode_pic(self.X, self.grid_dim, self.subgrid_dim, 
                                self.latent_dim, self.ae_model, device).reshape(-1)
            
        # Define state to be a subgrid of X and grid compressed z
        # Choose first subgrid at initialization to be in the left upper corner
        self.state[:self.change_dim] = np.copy(self.X[:self.subgrid_dim, :self.subgrid_dim]).reshape(-1)
        
        return self.state
    
                             