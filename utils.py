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

# Initialize grid value for direct mode
def initialize_direct():
    z = np.zeros((4, 4))
    z[1:4, 1:4] = np.random.rand(3, 3)
    z -= .5
    return z

# Spline geometry helper
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

def extract_geometry(img, return_contour=False):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0; peri = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)
    if not return_contour:
        return area, peri
    else:
        return area, peri, contours

def isoperi_reward(img, return_contour=False):
    if not return_contour:
        area, peri = extract_geometry(img)
        if peri == 0:
            return 0
        else:
            return np.sqrt(abs(area))/abs(peri)
    else:
        area, peri, contours = extract_geometry(img, True)
        if peri == 0:
            return 0, contours
        else:
            return np.sqrt(abs(area))/abs(peri), contours

def extract_geometry_from_array(z, xk=None, yk=None, xg=None, yg=None):
    if xk is None:
        xk, yk = np.mgrid[-1:1:z.shape[0]+0j, -1:1:z.shape[1]+0j]
    if xg is None:
        xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
    return extract_geometry(spline_interp(xk, yk, z, xg, yg))

# Initialization for big image
def initialize_single(grid_dim=8, subgrid_dim=16, num_milestone=17, spacing=16, mode='random', xg=None, yg=None):
    '''
    Initialize a shape with a single hierarchy
    '''
    if xg is None:
        xg, yg = np.mgrid[-1:1:50j, -1:1:50j]
    dimX = grid_dim * subgrid_dim
    X = None
    # Initialize X with different modes
    if mode=='random':
        X = np.random.rand(dimX, dimX)
    if mode=='mountain':
        """ 
        Choose some random number of up milestones and down milestones and interpolate to get 
        the up and down heights of the shape along the horizontal
        """
        half_dim = dimX/2
        X = np.zeros((dimX, dimX))
        milestoneX_up = np.zeros(num_milestone)
        milestoneX_down = np.zeros(num_milestone)
        for i in range(num_milestone):
            milestoneX_up = np.random.randint(low=half_dim/4, high=3*half_dim/4, size=num_milestone)
            milestoneX_down = np.random.randint(low=half_dim/4, high=3*half_dim/4, size=num_milestone)
        k = half_dim/2 + np.random.randint(low=-half_dim/4, high=half_dim/4)
        for i in range(num_milestone-1):
            for j in range(spacing):
                up = int(milestoneX_up[i]*(1.0-(j/spacing)) + milestoneX_up[i+1]*(j/spacing))
                down = int(milestoneX_down[i]*(1.0-(j/spacing)) + milestoneX_down[i+1]*(j/spacing))
                X[half_dim-down:half_dim+up, k+j] = np.ones(up+down)
            k += spacing
        X -= 0.5
        
    # return area and perimeter information on each cell of the large grid. Each cell correspond to a subgrid
    area, peri = np.zeros((grid_dim, grid_dim)), np.zeros((grid_dim, grid_dim))
    for i in range(grid_dim):
        for j in range(grid_dim):
            area[i][j], peri[i][j] = extract_geometry_from_array(\
                X[i*subgrid_dim:(i+1)*subgrid_dim, j*subgrid_dim:(j+1)*subgrid_dim], xg=xg, yg=yg)

    total_area, total_peri = np.sum(area), np.sum(peri)
    
    return area, peri, total_area, total_peri, X

import torch

def decode_pic(X, grid_dim, subgrid_dim, latent_dim, vae_model, device):
    """
    X is a grid with dimension grid_dim. Each cell in X grid is another subgrid with dim subgrid_dim
    For example if grid_dim = 8 and subgrid_dim = 16 then X actually has dimension 8*16 = 128
    """
    z = torch.tensor(np.zeros((latent_dim, grid_dim, grid_dim)), dtype=torch.float)
    for i in range(grid_dim):
        for j in range(grid_dim):
            mu, logvar = vae_model.encode(torch.tensor(X[i*subgrid_dim:(i+1)*subgrid_dim,\
                        j*subgrid_dim:(j+1)*subgrid_dim], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device))         
            z[:, i, j] = vae_model.reparameterize(mu, logvar).cpu().squeeze()
    return z                  

# Training VAE initialization
def get_sample(subgrid_dim, num_milestone, spacing):
    """
    Get meaningfully geometric samples for VAE pretraining

    """
    x = 0.5*np.random.rand(subgrid_dim, subgrid_dim)
    half_dim = subgrid_dim/2
    milestoneX_up = np.zeros(num_milestone)
    milestoneX_down = np.zeros(num_milestone)
    # random variable showing orientation
    orient = np.random.rand()
    # both top and bottom
    if orient < 0.2:
        for i in range(num_milestone):
            milestoneX_up = np.random.randint(low=half_dim/4, high=3*half_dim/4, size=num_milestone)
            milestoneX_down = np.random.randint(low=half_dim/4, high=3*half_dim/4, size=num_milestone)
        k = 0
        for i in range(num_milestone-1):
            for j in range(spacing):
                up = int(milestoneX_up[i]*(1.0-(j/spacing)) + milestoneX_up[i+1]*(j/spacing))
                down = int(milestoneX_down[i]*(1.0-(j/spacing)) + milestoneX_down[i+1]*(j/spacing))
                x[half_dim-down:half_dim+up, k+j] = 0.6+0.4*np.random.rand(up+down)
            k += spacing
    # only top, bottom, right or left:
    else:     
        milestone = np.random.randint(low=12, high=24, size=5)
        len_one = np.zeros(subgrid_dim).astype(int)
        for i in range(num_milestone-1):
            for j in range(spacing):
                len_one[spacing*i+j]= int(milestone[i]*(1.0-(j/spacing)) + milestone[i+1]*(j/spacing))
        # only top
        if orient < 0.4:
            for i in range(subgrid_dim):
                x[i,:len_one[i]] = 0.7+0.3*np.random.rand(len_one[i])
        # only bottom
        elif orient < 0.6:
            for i in range(subgrid_dim):
                x[i, subgrid_dim-len_one[i]:subgrid_dim] = 0.7+0.3*np.random.rand(len_one[i])
        # only left
        elif orient < 0.8:
            for i in range(subgrid_dim):
                x[:len_one[i], i] = 0.7+0.3*np.random.rand(len_one[i])
        # only right
        else:
            for i in range(subgrid_dim):
                x[subgrid_dim-len_one[i]:subgrid_dim, i] = 0.7+0.3*np.random.rand(len_one[i])

    return torch.tensor(x, dtype=torch.float)

def get_batch_sample(batch_size, subgrid_dim, num_milestone, spacing):
    """
    Get meaningfully geometric batches of samples for VAE pretraining

    """
    x = torch.zeros(batch_size, 1, 32, 32, dtype=torch.float)
    for i in range(batch_size):
        x[i,:,:,:] = get_sample(subgrid_dim, num_milestone, spacing)
    return x

# Update locally big image in single hierarchy setting
def update_state(X, i, j, x, area, peri, total_area, total_peri,
                 xk, yk, xg, yg):
    # Calculate new area and peri for the new updated portion
    new_area, new_peri = extract_geometry_from_array(x, xk, yk, xg, yg)
    # Update total_area, total_peri
    total_area += new_area-area[i,j]
    total_peri += new_peri-peri[i, j]
    # Update grid area and peri
    area[i][j], peri[i][j] = new_area, new_peri
    # Update x
    X[i*x.shape[0]:(i+1)*x.shape[0]][j*x.shape[1]:(j+1)*x.shape[1]] = x
    
    return total_area, total_peri
    
# FOR LATER USE
def subgrid_ind(p):
    """
    Find indices of the changing subgrid given changing quadrants vector p
    """
    cx, cy = 0, 0
    for i in range(4):
        p[i] = int(p[i])
        cx = cx*2 + (p[i]%2)
        cy = cy*2 + (p[i]//2)
    return int(cx), int(cy)
