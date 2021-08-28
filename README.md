# shape_vae

Apply reinforcement learning to very high dimensional dataset to solve optimization problem. Latent representation and multi-level (hierarchical) methods are used to handle high-dimensional data. 

The current data we're dealing with is 2D distributions: each datapoint is a 2D distribution. This type of data is used to represent shape (main mathematical object in engineering design problems) using level-set method.

There are currently three folders corresponding to different approaches we're working on:
* direct: Discretize 2D distribution by bicubic spline and use a direct network for optimization tasks.
* single_level: Use a latent representation to improve the training.
* multi_level: Use both latent representation and a hierarchy with multi-level network to efficiently apply reinforcement learning to optimization tasks.
