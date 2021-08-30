import numpy as np
import cv2

from utils import initialize, spline_interp, isoperi_reward
import gym
from gym import spaces
from gym.utils import seeding

#import torch

#device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

class IsoperiTestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # xk, yk: knot coordinates; xg, yg: grid coordinates
    def __init__(self, xk=None, yk=None, xg=None, yg=None, out_file='test_vid/test2.wmv'):
        super(IsoperiTestEnv, self).__init__()        
        self.num_step = 0
        self.max_val = 10; self.min_val = -10
        
        if xk is None:
            xk, yk = np.mgrid[-1:1:4j, -1:1:4j]
        if xg is None:
            xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
        
        self.num_coef = xk.shape[0]*yk.shape[1]
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
        
        # Create video writer
        self.out_file = out_file
        fourcc = cv2.VideoWriter_fourcc(*'WMV1')
        self.out = cv2.VideoWriter(self.out_file, fourcc, 20.0, (xg.shape[0], yg.shape[1]), isColor=False)
        # Other useful fields
        self.seed()
        self.reward = 0
        
        print('Done initialize')
    
    # For probabilistic purpose (unique seed for the obj). To be used.
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        # Update state (linearly)
        self.num_step += 1
        eps = 0.000002
        self.state += eps*act    
        # Interpolate main values with bicubic spline
        img = spline_interp(self.xk, self.yk, self.state.reshape(self.xk.shape[0], self.yk.shape[0]), self.xg, self.yg)
        # Update reward
        reward, contours = isoperi_reward(img, return_contour=True)
        
        # Check previous reward and update if things not improved
        if self.reward >= 1.1*reward:
            self.state -= eps*act
            act = np.random.rand(self.state.shape[0]) - 0.5
            self.state += eps*act
            # Interpolate main values with bicubic spline
            img = spline_interp(self.xk, self.yk, self.state.reshape(self.xk.shape[0], self.yk.shape[0]), self.xg, self.yg)
            # Update reward
            reward, contours = isoperi_reward(img, return_contour=True)
        
        self.reward = reward
        
        # Print info
        if self.num_step % 200 == 1:
            print('Reward: ', reward, '; Step: ', self.num_step)
        # Update frame for trajectory's video rendering
        if self.num_step % 200 == 1:
            cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0,255,0), thickness=1)
            self.out.write(img)
            
        # Stop when num of step is more than 2000
        done = False
        if self.num_step >= 40000:
            done = True
            # Save video
            self.out.release()
            
        return np.array(self.state), reward, done, {}

    def reset(self):
        return self.reset_at()
        
    def reset_at(self, shape='random'):
        self.num_step = 0
        num_contour = 0
        reward = 1
        while num_contour != 1 or (reward == 0 or reward >= 0.14):
            # Random z
            z = initialize()
            img = spline_interp(self.xk, self.yk, z, self.xg, self.yg)
            reward, contours = isoperi_reward(img, return_contour=True)
            num_contour = len(contours)
            self.state = z.reshape(-1)

        return self.state
    
                             