import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# Contains functions to evaluate VAEs based certain metrics
class Metric:
    
    def __init__(self, vae: nn.Module, dataset: torch.utils.data.Dataset, batch_size: int = 64):
        self.vae = vae
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=1, shuffle=False)

    '''
    # The mutual information gap (MIG)
    # Implmented according to the paper ()
    # 
    # Finds
    # 
    # PARAMS:
    #  
    def mutual_information_gap(self, factors):
        # Get dimensions
        num_examples = len(self.dataset)
        latent_dims = self.vae.latent_dims
        self.vae.eval()
        
        # Get mu, logvar, and a latent sample for each example in the dataset
        n = 0
        q_params = torch.Tensor(num_examples, latent_dims, 2)
        z_samples = torch.Tensor(num_examples, latent_dims)
        for x_batch in self.dataloader:
            bs = x_batch.size(0)
            mu, logvar = self.vae.encode(x_batch)
            z = self.vae.reparameterize(mu, logvar)
            z_samples[n:n+bs, :] = z
            q_params[n:n+bs, :, 0] = mu
            q_params[n:n+bs, :, 1] = logvar
            n += bs
        
        # Calculate marginal entropy
        
        return
    '''
    
    # The KNN-MSE metric tests the local coherence of the state representation
    # Implemented according to the paper 
    # Unsupervised state representation learning with robotic priors: a robustness benchmark (https://arxiv.org/pdf/1709.05185.pdf)
    # 
    # Finds KNN-MSE = \frac{1}{N}\sum_{z_i}^N \frac{1}{k} \sum_{z_j \in \text{KNN}(z_i, k)}^k (z_i - z_j)^2
    # 
    # PARAMS:
    # num_neighbors (int): the number of neighbors for each data point excluding itself
    def knn_mse(self, num_neighbors):
        if self.vae.training:
            raise Exception("The vae must be in eval mode. Call vae.eval() before this function")
        # Encode all images in the dataset
        encoded_states = []
        total_knn_dist = 0
        with torch.no_grad():
            # Encode each input in the dataset into z 
            for x, _ in tqdm(self.dataset):
                mu, logvar = self.vae.encode(torch.unsqueeze(x, 0))
                encoded_state = self.vae.reparameterize(mu, logvar)
                encoded_states.append(torch.squeeze(encoded_state).numpy())
            encoded_states = np.array(encoded_states)
            # Find the k nearest neighbors of all the states (i.e. KNN(I, k))
            neigh = NearestNeighbors(n_neighbors=num_neighbors).fit(encoded_states)
            # Calculate sum of Euclidean distances between each data point and its neighbors in the latent space 
            total_knn_dist= neigh.kneighbors_graph(mode="distance").sum() # \sum_{z_i} \sum_{z_j \in \text{KNN}(z_i, k)} (z_i - z_j)^2 
        # Calculate the KNN-MSE error    
        total_knn_mse = total_knn_dist / (num_neighbors * len(self.dataset)) # \frac{1}{kN}\sum_{z_i} \sum_{z_j \in \text{KNN}(z_i, k)} (z_i - z_j)^2
        return total_knn_mse

    # Finds the entropy of the latent space
    # 
    # PARAMS:
    # z_samples (Tensor of Size[latent_dims, num_examples]): a tensor containing a latent sample for every example in the dataset
    # qz_parameters (Tensor of Size[num_examples, latent_dims, num_params]): a tensor containing mu_z and sigma_z for every example in the dataset
    # num_samples (int): Number of times to sample the input distribution p(x)
    def estimate_entropy(self, z_samples, qz_parameters, num_samples=10000):        
        # Sample p(x) by selecting num_samples random indices into the dataset
        x_indices = torch.randperm(z_samples.shape[1])[:num_samples].cuda()
        # Sample q(z|x)
        z_samples = z_samples.index_select(1, x_indices)

        latent_dims, num_z_samples = z_samples.shape
        num_examples, _, num_params = qz_parameters.shape
        # 
        weights = -math.log(num_examples)

        entropies = torch.zeros(latent_dims).cuda()
        # Broadcast the first dimension of z_samples and qz_parameters to num_examples
        z_samples =  z_samples.view(1, latent_dims, num_z_samples).expand(num_examples, latent_dims, num_z_samples)
        qz_parameters = qz_paramameters.view(num_examples, latent_dims, 1, num_params).expand(num_examples, latent_dims, num_z_samples, num_params)
        # Compute entropy for each sample
        # -log(\sum exp(log_qz_i - log(num_examples)))
        while k < num_z_samples:
            # Clip batch_size if a batch does not fit at the end
            bs = min(self.batch_size, num_z_samples-k)
            # Compute log density of latent distribution of each sample(i.e. log_qz_i)
            log_qz_i = log_density(
               z_samples[:, :, k:k + bs],
               qz_parameters[:, :, k:k + bs] 
            )
            entropies += -logsumexp(log_qz_i + weights, dim=0, keepdim=False).data.sum(1)
            k += bs
        return entropies / latent_dims

    # The NIEQA metric measures the local and glovbla neighborhood embedding quality 
    # TODO
    def nieqa_metric(self):
        return

    '''
    THE HIGGINS METRIC IS NOT CONSIDERED VERY ACCURATE DUE TO THE VARIABILITY OF THE METRIC DUE TO HYPERPARAMETERS
    # A low VC-dimensuon linear classifier is used to predict the image based
    # on a truth factor (i.e. color, size)
    # 
    # PARAMS:
    # inputs_size (int):
    # output_size (int):
    # factor ():
    def higgins_metric(self, input_size: int, output_size: int, factor):
        # Set up linear classifier network
        linear_classifier = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Softmax()
            )
        loss = nn.NLLLoss()
        return
    '''


# Calculate the log density of a Gaussian in a numerically stable way
def log_density(x, mu, logvar):
    normalization = torch.Tensor([np.log(2 * np.pi)]) # log(2pi)
    inv_sigma = torch.exp(-logvar) # 1/sigma
    tmp = (x - mu) * inv_sigma # (x-mu)/sigma
    return -0.5 * (tmp * tmp + 2 * logvar + normalization) # -0.5(((x-mu)/sigma))^2 + 2*log(sigma) + log(2pi)

# Calculates the logsumexp in a numerically stable way
# 
# Finds log(sum(exp(value)))
def logsumexp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)