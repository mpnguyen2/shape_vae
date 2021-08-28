#!/usr/bin/env python
# coding: utf-8

# In[177]:


import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


theta_cache = {}

def create_sample(num_frames=300, plot=False):
    # Input constants 
    m = 1 # mass (kg)
    L = 1 # length (m)
    b = 2 # damping value (kg/m^2-s)
    g = 9.81 # gravity (m/s^2)
    delta_t = 0.1 # time step size (seconds)
    t_max = 5 # max sim time (seconds)
    theta1_0 = np.random.rand() * 2 * np.pi
    theta2_0 = np.random.uniform(-2*np.pi, 2*np.pi) # initial angular velocity (rad/s)
    theta_init = (theta1_0, theta2_0)
    # Get timesteps
    t = np.linspace(0, t_max, t_max/delta_t)
    
    def int_pendulum_sim(theta_init, t, L=1, m=20, b=5, g=9.81):
        theta_dot_1 = theta_init[1]
        theta_dot_2 = -b/m*theta_init[1] - g/L*np.sin(theta_init[0])
        return theta_dot_1, theta_dot_2

    
    def draw_circle(x, y, val, data, r=15):
        for i in range(280):
            for j in range(280):
                if((x-i)**2+(y-j)**2 <= r**2):
                    data[j][i] = val
    def draw_line(x0,y0,x1,y1, val, data):
        
        def close_to(a,b):
            return abs(a-b) <=10
        
        m = (y1-y0)/(x1-x0)
        print(m)
        for i in range(280):
            for j in range(280):
                if(close_to(j - y1,m*(i - x1))):
#                     print("YAHOO", m)
                    data[j][i] = 2

    def downsize(data, width, height):
        data_prime = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                # get average of 100 pixels corresponding to i,j
                new_j = j*10
                new_i = i*10
    #             print(new_j, new_j+10, new_i, new_i+10)
                submatrix = data[new_j: new_j+10, new_i: new_i+10]
    #             plt.imshow(submatrix)
    #             plt.show()
                data_prime[j][i] = submatrix.mean()
        return data_prime
    
    def make_image(img_num, theta):
        # check cache
        def to_index(theta):
            return round(theta % (2*np.pi),1)
#         print(to_index(theta))
        if(to_index(theta) in theta_cache):
#             print("CACHE HIT: ", theta, to_index(theta))
            return theta_cache[to_index(theta)]
        # otherwise, create new image
        L = 100

        w = 280
        h = 280
        data = np.zeros((w,h))
        
        # origin/fulcrum
        y0 = w//2
        x0 = h//2


        y = L * np.cos(theta) + y0
        x = L * np.sin(theta) + x0

        draw_circle(x0,y0,val=1,data=data)
        draw_circle(x,y,val=1,data=data, r = 25)
    #     print(data.shape)
#         draw_line(x0, y0, x, y, val=1, data=data)
        data_prime = downsize(data, 28, 28)
#         plt.imshow(data_prime, cmap='gray')
#         plt.savefig(str(img_num)+'.png')
#         plt.show()
        theta_cache[to_index(theta)]= data_prime.flatten()
        return data_prime.flatten()
    
    theta_vals_int = integrate.odeint(int_pendulum_sim, theta_init, t)
    pos_thetas = theta_vals_int[:,0]
    plt.plot(t, theta_vals_int)

    sequence = np.stack(list(make_image(i, pos_thetas[i]) for i in range(len(pos_thetas))))
    return sequence


# In[218]:


def create_data_matrix(num_samples, num_frames):
    X = (create_sample(num_frames) for i in range(num_samples))
    X_prime = np.stack(list(X))
    return X_prime

X = create_data_matrix(num_samples=1, num_frames = 50)


# In[172]:


np.save('pendulum.npy', X)    # .npy extension is added if not given


# In[219]:


for img in X[0]:
    plt.imshow(img.reshape(28,28),cmap='gray')
    plt.savefig(str(i)+'.png')
    i += 1


# In[ ]:




