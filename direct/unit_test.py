import numpy as np
import torch

import scipy

from torch.distributions import Categorical, multivariate_normal
#from net_model import LatentRL
import cv2
import torch.nn.functional as F
from net_model import AE
from utils import reward, get_batch_sample, get_reward_sample, initializeAE

import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from scipy import interpolate
import matplotlib.pyplot as plt
import time

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
'''

start_time = time.time()

x, y = np.mgrid[-1:1:4j, -1:1:4j]
z = np.zeros((4, 4))
z[1:4, 1:4] = np.random.rand(3, 3)
#z = np.random.rand(x.shape[0], x.shape[1])
z -= 0.5
#z = np.random.rand(x.shape[0], x.shape[1]) - .5

xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
tck = interpolate.bisplrep(x, y, z)
zint = interpolate.bisplev(xg[:,0], yg[0,:], tck)

'''
# Using object
x = np.arange(-1, 1, 1/4)
y = np.arange(-1, 1, 1/4)
z = np.random.rand(x.shape[0], y.shape[0]) - .5

obj = scipy.interpolate.RectBivariateSpline(x, y, z)
xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
zint = obj.ev(xg, yg, dx=1, dy=1)
'''

draw = True
if draw:
    plt.figure()
    lims = dict(cmap='RdBu_r', vmin=-.5, vmax=.5)
    plt.pcolormesh(y, -x, z, shading='auto', **lims)
    plt.colorbar()
    plt.title("Sparsely sampled function.")
    plt.show()
    
    plt.figure()
    lims = dict(cmap='RdBu_r', vmin=-.5, vmax=.5)
    plt.pcolormesh(yg, -xg, zint, shading='auto', **lims)
    plt.colorbar()
    plt.title("Cubic spline function.")
    plt.show()

'''
zint = np.zeros((100, 100))
zint[25:75, 25:75] = np.ones((50, 50))
zint[40:45, 40:45] = np.zeros((5, 5))
zint[60:65, 60:65] = np.zeros((5, 5))
zint -= 0.5
'''

C = 255/2

#get threshold image
#cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = np.array(zint*C+C).astype('uint8')
#plt.imshow(img, cmap='gray')
thresh = C
ret,thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = np.zeros(xg.shape)
# draw the contours on the empty image
cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0,255,0), thickness=1)

area = 0; peri = 0
for cnt in contours:
    area -= cv2.contourArea(cnt, oriented=True)
    peri += cv2.arcLength(cnt, closed=True)

if draw:
    plt.imshow(img, cmap='gray')

# Print stats
print('Number of contours: ', len(contours))
print('Area: ', area)
print('Perimeter: ', peri)
print('Runtime: ', time.time()-start_time)

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