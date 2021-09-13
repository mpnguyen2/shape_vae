import numpy as np
import torch

device = 'cpu'

def policy_gradient(ac_opt, ac_model, gamma, T, episode, b):
    """
    Train the actor-critic-artificial reward net for one episode
    """
    p_policy_losses = [] #list to save actor loss for position
    v_policy_losses = [] #list to save actor loss for the change amount v
    value_losses = [] #list to save critic loss
    
    # Calculate cumulative reward R backwardly
    cum_rewards = []
    R = np.zeros(b)
    for r in ac_model.rewards[:]:
        R = r + gamma*R
        cum_rewards.insert(0, R)
    
    # Tensorize and normalize cumulative rewards
    cum_rewards = torch.tensor(cum_rewards).to(device)
    #cum_rewards = (cum_rewards-cum_rewards.mean(dim=0))/(cum_rewards.std(dim=0) + eps)
    
    # Calculate individual terms in policy and critic loss fcts
    for (p_logprob, v_logprob, value), R in zip(ac_model.saved_actions, cum_rewards):
        advantage = R.unsqueeze(1) # if actor have to subtract value.item()
        # calculate policy losses
        #print(-v_logprob*advantage)
        p_policy_losses.append(-p_logprob*advantage)
        v_policy_losses.append(-v_logprob*advantage)
        # calculate critic losses
        # value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
        
    ac_opt.zero_grad()
    
    # sum up all loss
    p_loss = torch.stack(p_policy_losses).sum()
    v_loss = torch.stack(v_policy_losses).sum()
    loss = p_loss + v_loss

    #value_losses = torch.stack(value_losses).mean()
    # Discourage similar values to escape local min
    #loss -= torch.log(F.mse_loss(torch.stack(values[:-1]), torch.stack(values[1:])) + eps).sum()
    
    #reward_loss = F.mse_loss(torch.tensor(ac_model.rewards),\
    #            torch.stack(ac_model.net_rewards).squeeze(), reduction='mean')
    reward_loss = 0
    
    # grad descent
    loss.backward()      
    ac_opt.step()
    
    log_reward = ac_model.rewards[-1].mean().item()

    
    del ac_model.rewards[:]
    del ac_model.saved_actions[:]
    del ac_model.net_rewards[:]
    
    return log_reward, reward_loss