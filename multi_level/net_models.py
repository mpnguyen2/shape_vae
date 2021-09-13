""" 
This file consists of description of all multi-hierarchy network
"""

import torch
from torch import nn
from torch.nn import functional as F

#from torchsummary import summary
#from torch.utils.tensorboard import SummaryWriter

#print(torch.__version__)

class AC_Multilevel(nn.Module):
    #TBA
    def __init__(self):
        super(AC_Multilevel, self).__init__()
        
    def forward(self, x):
        return x