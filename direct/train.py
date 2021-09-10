import torch

from stable_baselines3 import SAC, PPO


from net_models import ShapePPOPolicy
from isoperi import IsoperiEnv
from isoperi_test import IsoperiTestEnv


# Fixed global vars
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

# Default is SAC 
timestep = 2000000; learning_rate = 3e-4; log_interval = 50
training_sac, training_ppo = False, False
testing_sac, testing_ppo = False, True

## TRAINING ##
if training_sac:
    # Create latent env
    env = IsoperiEnv()
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                     net_arch=dict(pi=[16, 32, 16, 8, 16, 8, 4, 4], qf=[32, 16, 16, 8, 4, 8, 4]))
    model = SAC("MlpPolicy", policy_kwargs=policy_kwargs, env=env, verbose=1, learning_rate=learning_rate)
    model.device = device
    model.learn(total_timesteps=timestep, log_interval=log_interval)
    model.save("models/sac_shape_opt")

if training_ppo:
    # Create latent env
    env = IsoperiEnv()
    # Create agent and learn
    model = PPO(policy=ShapePPOPolicy, env=env, verbose=1, learning_rate=learning_rate)
    model.learn(total_timesteps=timestep, log_interval=log_interval)
    model.save("models/ppo_shape_opt")

## TESTING ##
if testing_sac:
    # Create test env
    env = IsoperiTestEnv()
    
    # Loading model
    model = SAC.load("models/sac_shape_opt")
    model.set_env(env)
    
    obs = env.reset()
    done = False
    while done == False:
        action, _states = model.predict(obs)
        #action = np.random.rand(16)-.5
        obs, rewards, done, info = env.step(action)
        #env.render()

if testing_ppo:
    # Create test env
    env = IsoperiTestEnv()
    
    # Loading model
    model = PPO.load("models/ppo_shape_opt")
    model.set_env(env)
    
    obs = env.reset()
    done = False
    while done == False:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        #env.render()