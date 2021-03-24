"""
Testing files.
"""
from train import train
'''
Do some testing here
'''

#### TESTING TRAIN ####

train(vae_training_size=1024, vae_num_epoch=4001, num_eps_with_VAE=400, \
      num_eps_without_VAE=0, num_traj=4, initialize_interval=10,\
          T = 64, discounted_rate = 0.99, log_interval=10)
