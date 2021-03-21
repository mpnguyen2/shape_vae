import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# Contains functions to evaluate VAEs based certain metrics
class Metric:
    
    def __init__(self, vae, dataloader, dataset):
        self.vae = vae
        self.dataloader = dataloader
        self.dataset = dataset

    
    # A low VC-dimensuon linear classifier is used to predict the image based
    # on a truth factor (i.e. color, size)
    # 
    # PARAMS:
    # inputs_size (int):
    # output_size (int):
    # factor ():
    def higgins_metric(self, input_size, output_size, factor):
        # Set up linear classifier network
        linear_classifier = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Softmax()
            )
        loss = nn.NLLLoss()
        return

    # Calculates the mutual information gap
    def mutual_information_gap(self):
        return
    
    # The KNN-MSE metric tests the local coherence of the state representation
    # Implemented according to the paper (https://arxiv.org/pdf/1709.05185.pdf)
    # 
    # PARAMS:
    # num_neighbors (int): the number of neighbors for each data point excluding itself
    # x (tensor):  Tensor containing the 
    # 
    def knn_mse(self, num_neighbors):
        if self.vae.training:
            raise Exception("The vae must be in eval mode. Call vae.eval() before this function")
        
        # Encode all images in the dataset
        encoded_states = []
        total_knn_dist = 0
        with torch.no_grad():
            for x, _ in tqdm(self.dataset):
                mu, logvar = self.vae.encode(torch.unsqueeze(x, 0))
                encoded_state = self.vae.reparameterize(mu, logvar)
                encoded_states.append(torch.squeeze(encoded_state).numpy())
            encoded_states = np.array(encoded_states)
            # Find the k nearest neighbors of all the states (i.e. KNN(I, k))
            neigh = NearestNeighbors(n_neighbors=num_neighbors+1).fit(encoded_states)
            # Calculate KNN-MSE error
            for encoded_state in encoded_states:
                neigh_dists, _ = neigh.kneighbors(encoded_state.reshape(1, -1))
                total_knn_dist += neigh_dists.sum()
            graph = neigh.kneighbors_graph(mode="distance")
            print(graph)
            print(graph.sum())
            print(total_knn_dist)
        total_knn_mse = total_knn_dist / (num_neighbors * len(self.dataset))
        return total_knn_mse
        
    # The NIEQA metric measures the local and glovbla neighborhood embedding quality 
    def nieqa_metric(self):
        return
    
    # Axis aligned metric
    def axis_aligned_metric(self):
        return

def estimate_entropy():
    return