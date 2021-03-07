""" 
This file consists of helper functions 

"""

""" Again: X is the giant image but each time one small part of it is modified
"""

def initialize(X):
""" Initialize a shape (represented by bitmap images of dim 512*512)
"""

def step(z, p, v):
""" Simulate the dynamics of latent state
    and output both the true reward
p: position of the subgrid to be updated
v: how much to update
"""
"""
    return next_z1
"""

def updateX(X, r, p, x):
""" Calculate new reward from old reward using only part is modified
and neighboring cells in X
X: big image
r: old reward
p: position of the subgrid to be updated
x: the updated new subgrid
"""
"""
   Modified x
   return r
"""

def updateZ(z, p, v):
    