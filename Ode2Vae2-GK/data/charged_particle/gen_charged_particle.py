#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Source: https://flothesof.github.io/charged-particle-trajectories-E-and-B-fields.html


# In[8]:


# imports
import numpy as np
from scipy.integrate import ode


# In[9]:


def newton(t, Y, q, m, B, E):
    """Computes the derivative of the state vector y according to the equation of motion:
    Y is the state vector (x, y, z, u, v, w) === (position, velocity).
    returns dY/dt.
    """
    x, y, z = Y[0], Y[1], Y[2]
    u, v, w = Y[3], Y[4], Y[5]
    
    alpha = q / m 
    return np.array([u, v, w, 0, alpha * B* w + E, -alpha * B * v])


# In[10]:


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


# In[29]:


def create_sample(num_frames):
    t0 = 0
    x0 = np.array([np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-10,10)])
    v0 = np.array([np.random.uniform(-10,10), np.random.uniform(-10,10), np.random.uniform(-10,10)])
    initial_conditions = np.concatenate((x0, v0))
    
    r = ode(newton).set_integrator('dopri5')
    
    r.set_initial_value(initial_conditions, t0).set_f_params(1.0, 1.0, 1.0, 10.)
    positions = []

    dt = 0.1
    
    t1 = num_frames * dt
        
        
    while r.successful() and r.t < t1:
        r.integrate(r.t+dt)
        positions.append(r.y[:3])

    positions = np.array(positions)
    
    return positions


# In[30]:


positions = create_sample(200)
positions.shape


# In[31]:


def create_data_matrix(num_samples, num_frames):
    X = (create_sample(num_frames) for i in range(num_samples))
    X_prime = np.stack(list(X))
    return X_prime

X = create_data_matrix(num_samples=1042, num_frames = 200)


# In[32]:


X.shape


# In[48]:


np.save("charge_particle.npy",X)


# In[61]:


from mpl_toolkits.mplot3d import Axes3D

def plot_example_sequence(example_sequence):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(example_sequence[:, 0],example_sequence[:, 1], example_sequence[:, 2])

    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    
plot_example_sequence(X[9])
plot_example_sequence(X[3])

