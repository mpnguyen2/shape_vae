import numpy as np
import torch
from torch.distributions import Categorical, multivariate_normal
#from net_model import LatentRL
import cv2


img = cv2.imread('initial_images/heart.bmp')
img = img[:,:,1]
img = np.round(img/255)
print(img.sum())
print(img[12:24, 12:24])

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