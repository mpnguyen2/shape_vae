import numpy as np
import torch
from torch.distributions import Categorical, multivariate_normal
#from net_model import LatentRL
import cv2
import torch.nn.functional as F
from net_model import AE
from utils import reward, get_batch_sample, get_reward_sample, initializeAE

import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
#device = torch.device("cpu")
'''
img = cv2.imread('initial_images/circle.bmp')
img = img[:,:,1]
img = 1-np.round(img/255)
print(img[0:10, 0:10])
'''
'''
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
'''
ae_model_path = 'models/ae_model1'
#reward_model_path = 'models/reward_model1'

# Test reward
# Build and load ae and reward models
ae_model = AE().to(device)
#reward_model = Reward().to(device)
#ae_model.load_state_dict(torch.load(ae_model_path))
#reward_model.load_state_dict(torch.load(reward_model_path))

x = torch.tensor(np.random.rand(1, 32, 32), dtype=torch.float).to(device)
x1, z = ae_model(x)
print(x1.shape, z.shape)

# Test and compare
#z, R = get_reward_sample(ae_model, 10000, device)
#print(np.max(R.cpu().detach().numpy()))
#for i in range(50):
#    print(R[100*i])

#cv2.waitKey(0) # wait for ay key to exit window
#cv2.destroyAllWindows()
'''
### TEST CUDA ### -> Using CUDA for VAE is better, not sure about ac net.
# Test AC
print('Test AC net:\n')
ac_model = LatentRL()
input = torch.rand(10, 8, 16, 16, dtype = torch.float)
cur_time = time.time()
output = ac_model(input)
print('Without cuda this takes {} seconds\n'.format(time.time()-cur_time))
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print('Device available: {}\n'.format(device))
ac_model.to(device)
input = input.to(device)
cur_time = time.time()
output = ac_model(input)
print('With cuda this takes {} seconds\n'.format(time.time()-cur_time))

print(torch.cuda.device_count())

cur_time = time.time()
output = ac_model(input)
print('With cuda parallel this takes {} seconds\n'.format(time.time()-cur_time))

# Test VAE
print('Test VAE net:\n')
vae_model = VAE()
input = torch.rand(128, 1, 32, 32, dtype = torch.float)
cur_time = time.time()
output = vae_model(input)
print('Without cuda this takes {} seconds\n'.format(time.time()-cur_time))
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print('Device available: {}\n'.format(device))
vae_model.to(device)
input = input.to(device)
cur_time = time.time()
output = vae_model(input)
print('With cuda this takes {} seconds'.format(time.time()-cur_time))
'''