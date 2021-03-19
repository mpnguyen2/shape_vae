import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# Contains functions to evaluate VAEs based certain metrics
class Metric:
    
    def __init__(self, vae):
        return
    
    # A low VC-dimensuon linear classifier is used to predict the image based
    # on a truth factor (i.e. color, size)
    def higgins_metric(self, input_size, output_size):
        # Set up linear classifier network
        linear_classifier = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Softmax()
            )
        loss = nn.NLLLoss()
        

        # 
        return

    # Calculates the mutual information gap
    def mutual_information_gap(self):
        return
    
    # Computes the k-nearest neighbors
    def knn(self):
        return
    
    def knn_mse(self):
        return
    
    def nieqa_metric(self):
        return

    def axis_aligned_metric(self):
        return

def estimate_entropy():
    return