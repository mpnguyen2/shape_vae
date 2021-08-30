import numpy as np
from scipy import interpolate
import cv2

def gen_sigmoid(x, k, x0):
    return 1/(1 + np.exp(-k*(x-x0)))

def binary_reward(x_input, k = 10, x0 = .5):
    x = gen_sigmoid(x_input, k, x0)
    area = np.sum(x)
    peri = 4*area - 2*(np.sum(x[1:,:]*x[:-1,:]) \
                       + np.sum(x[:,1:]*x[:,:-1]))
    return np.sqrt(area)/peri

def initialize():
    z = np.zeros((4, 4))
    z[1:4, 1:4] = np.random.rand(3, 3)
    z -= .5
    return z

def spline_interp(xk, yk, z, xg, yg):
    # Interpolate knots with bicubic spline
    tck = interpolate.bisplrep(xk, yk, z)
    # Evaluate bicubic spline on (fixed) grid
    zint = interpolate.bisplev(xg[:,0], yg[0,:], tck)
    # zint is between [-1, 1]
    zint = np.clip(zint, -1, 1)
    # Convert spline values to binary image
    C = 255/2; thresh = C
    img = np.array(zint*C+C).astype('uint8')
    # Thresholding give binary image, which gives better contour
    _, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    return thresh_img

def isoperi_reward(img, return_contour=False):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0; peri = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)
    if peri == 0:
        if return_contour:
            return 0, contours
        return 0
    if return_contour:
        return np.sqrt(abs(area))/abs(peri), contours
    return np.sqrt(abs(area))/abs(peri)