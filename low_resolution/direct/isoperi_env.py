import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import torch
from utils import initialize, reward, square
from torch.utils.data import DataLoader, RandomSampler
import torchvision
import torchvision.transforms as transforms
from net_model import AE

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
batch_size=1
dataset = torchvision.datasets.MNIST(
    root='MNIST/data', train=False, download=False, transform=transform
)
dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

class IsoperiEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ae_model):
        super(IsoperiEnv, self).__init__()
        
        self.num_step = 0
        self.max_val = 10; self.min_val = -10
        #self.max_act = 1; self.min_act = -1
        self.num_coef = 16
        # action is now just the direction to move (discrete now)
        self.action_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.num_coef, ), dtype=np.float32) 
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.num_coef, ), dtype=np.float32) 
        # state is the level-set function phi
        self.state = None
        self.num_step = 0
        # pretrained model for latent environment use
        self.ae_model = ae_model

        # Other useful fields
        self.seed()
    
    # For probabilistic purpose (unique seed for the obj). To be used.
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, act):
        self.num_step += 1
        eps = 0.02
        self.state += eps*act

        #self.state = np.clip(self.state, 0.1, 0.9)
        # Find reward by using reward of decoder
        x_recon = self.ae_model.decode(torch.tensor(self.state, dtype=torch.float).to(device).unsqueeze(0))
        reward1 = reward(x_recon.squeeze().cpu().detach().numpy())
        #self.reward_model(torch.tensor(self.state, dtype=torch.float).to(device)).item()
        # When num of step is more than 200, stop
        done = False
        if self.num_step >= 100:
            done = True
        return np.array(self.state), reward1, done, {}

    def reset(self):
        return self.reset_at()
        
    def reset_at(self, shape='random'):
        self.num_step = 0
        '''
        #area, peri, x = initialize(device)
        x = torch.tensor(square(), dtype=torch.float).to(device)
        _, z = self.ae_model(x.unsqueeze(0))
        self.state = z.squeeze().cpu().detach().numpy()
        '''
        x, label= next(iter(dataloader))
        #while label.item() != 8:
        #    x, label= next(iter(dataloader))
        mu, logvar = self.ae_model.encode(x.to(device))
        self.state = mu.cpu().detach().numpy().squeeze()
    
        return self.state
    
                             