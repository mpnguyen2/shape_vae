#!/usr/bin/env python
# coding: utf-8

# In[46]:


# Source: https://matplotlib.org/3.3.2/gallery/mplot3d/lorenz_attractor.html

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz(x, y, z, s=10, r=28, b=2.667):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def create_sample(num_steps=300, plot=False):

    dt = 0.01
    

    # Need one more for the initial values
    xs = np.empty(num_steps)
    ys = np.empty(num_steps)
    zs = np.empty(num_steps)

    # Set initial values
    low = -10
    high = 10
    xs[0], ys[0], zs[0] = np.random.uniform(low, high, 3)
#     print(xs[0],ys[0],zs[0])
#     xs[0], ys[0], zs[0] = (0., 1., 1.05)

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps-1):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    
    if(plot):

        # Plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(xs, ys, zs, lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")

        plt.show()
    return np.stack((xs, ys, zs)).T

def create_data_matrix(num_samples, num_frames):
    X = (create_sample(num_frames) for i in range(num_samples))

    X_prime = np.stack(list(X))
    return X_prime

X = create_data_matrix(num_samples=1, num_frames = 200)

np.save('lorenz.npy', X)


# In[48]:


from mpl_toolkits.mplot3d import Axes3D

def plot_example_sequence(example_sequence):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(example_sequence[:, 0],example_sequence[:, 1], example_sequence[:, 2])

    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    
plot_example_sequence(X[0])


# In[ ]:




