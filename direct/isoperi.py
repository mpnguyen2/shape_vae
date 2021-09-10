import numpy as np
import cv2

import gym
from gym import spaces
from gym.utils import seeding

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   
from common.utils import initialize_direct, spline_interp, isoperi_reward


#device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class IsoperiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # xk, yk: knot coordinates; xg, yg: grid coordinates
    def __init__(self, xk=None, yk=None, xg=None, yg=None):
        super(IsoperiEnv, self).__init__()        
        self.num_step = 0
        self.max_val = 10; self.min_val = -10
        if xk is None:
            xk, yk = np.mgrid[-1:1:4j, -1:1:4j]
        if xg is None:
            xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
            
        self.num_coef = xk.shape[0]*yk.shape[0]
        # action is now just the direction to move (discrete now)
        self.action_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.num_coef, ), dtype=np.float32) 
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.num_coef, ), dtype=np.float32) 
        # state is the level-set function phi
        self.state = None
        self.num_step = 0
        # knot and fix grid
        self.xk = xk
        self.yk = yk
        self.xg = xg
        self.yg = yg

        # Other useful fields
        self.seed()
        
        # First reward
        self.reward = 1
    
    # For probabilistic purpose (unique seed for the obj). To be used.
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        # Update state (linearly)
        self.num_step += 1
        eps = 0.001
        self.state += eps*act    
        # Interpolate main values with bicubic spline
        img = spline_interp(self.xk, self.yk, self.state.reshape(self.xk.shape[0], self.yk.shape[0]), self.xg, self.yg)
        # Update reward
        reward = isoperi_reward(img)
        # Stop when num of step is more than 100
        done = False
        if self.num_step >= 100 or (self.num_step >= 8 and reward <= self.reward):
            done = True
    
        if self.num_step == 1:
            self.reward = reward
            
        return np.array(self.state), reward, done, {}

    def reset(self):
        return self.reset_at()
        
    def reset_at(self, shape='random'):
        self.num_step = 0
        num_contour = 0
        while num_contour != 1:
            # Random z
            z = initialize_direct()
            img = spline_interp(self.xk, self.yk, z, self.xg, self.yg)
            contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            num_contour = len(contours)
            self.state = z.reshape(-1)
    
        return self.state
    
                             