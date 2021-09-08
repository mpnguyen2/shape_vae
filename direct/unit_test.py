import numpy as np
import torch

import scipy

from torch.distributions import Categorical, multivariate_normal
#from net_model import LatentRL
import cv2

from scipy import interpolate
import matplotlib.pyplot as plt
import time

start_time = time.time()

x, y = np.mgrid[-1:1:4j, -1:1:4j]
z = np.random.rand(x.shape[0], x.shape[1])
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