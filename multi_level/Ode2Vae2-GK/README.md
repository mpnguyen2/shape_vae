# ODE<sup>2</sup>VAE++
#### Final project for Fall 2020 CS 378 Geometric Foundations of Data Science.

##### Authors: Grant Kluber <gkluber@utexas.edu> and Jonathan Randall <jonathan.randall@utexas.edu>.

This repository is a fork of ODE<sup>2</sup>VAE that attempts to improve and upgrade the original model. Upgraded files:
- train_ode2vae.py -- the upgraded version of the original training script
- test_ode2vae.py -- the upgraded version of the original testing script
- models/ode2vae_tfv1.py -- the upgraded version of the original uniform model
- models/data/* -- scripts for parsing the types of data, mostly unchanged

In addition, we have attempted to rewrite ODE<sup>2</sup>VAE from scratch in Tensorflow 2.x. The relevant files are:
- train.py and test.py -- rewritten training and testing scripts, with varying degrees of completeness
- models/ode2vae_tfv2.py -- simplest possible rewriting of ODE<sup>2</sup>VAE, but it still doesn't work
- util.py -- utilities primarily for Google absl-py

Our datasets are:
- Pendulum -- consists of 28x28 2D pendulum videos: N = 1046, T = 50, D = 784
- Lorenz attractor -- single particles in Lorenz attractor: N = 1046, T = 200, D = 3
- Electromagnetic field -- single charged particles: N = 1046, T = 200, D = 3