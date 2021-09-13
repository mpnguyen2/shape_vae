# File containing common types of networks. More specialized networks are in other files.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    """
    Simple multi-layer perceptron net (densly connected net)
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        layer_dims (List[int]): Dimensions of hidden layers
        activation (str): type of activations. Not applying to the last layer 
    """
    def __init__(self, input_dim, output_dim, layer_dims=[], activation='tanh'):
        super(Mlp, self).__init__()
        self.layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        if len(layer_dims) != 0:
            self.layers.append(nn.Linear(input_dim, layer_dims[0]))
            for i in range(len(layer_dims)-1):
                if activation == 'tanh':
                    self.layers.append(nn.Tanh())
                elif activation == 'relu':
                    self.layers.append(nn.ReLU())
                self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'relu':
                    self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(layer_dims[-1], output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, output_dim))
        # Composing all layers
        self.net = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.net(x)

class CNNEncoder(nn.Module):
    """
    A CNN encoder that map 3D tensor to 1D tensor (1D can be compressed or full info)
    Each time image size is cut by half
    Args:
        channels (List[int]): list of number of channels to be applied starting with original number of channels of 3D tensor input
        activation (str): type of activations. 
    """
    def __init__(self, channels, activation='tanh'):
        super(CNNEncoder, self).__init__()
        self.layers = []
        for i in range(len(channels)-1):
            self.layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            if activation == 'tanh':
                    self.layers.append(nn.Tanh())
            elif activation == 'relu':
                self.layers.append(nn.ReLU())
        # Composing all convolutional layers
        self.net = nn.Sequential(*self.layers, nn.Flatten())
    
    def forward(self, x):
        return self.net(x)
    
# Opposite to encoder
class CNNDecoder(nn.Module):
    """
    A CNN decoder that map 1D tensor to 3D tensor (1D can be compressed or full info)
    Each time image size is cut by half
    Args:
        channels (List[int]): list of number of channels to be applied starting with original number of channels of 1D tensor
        img_dim (int): width-height dimension of 3D tensor
        activation (str): type of activations. 
    """
    def __init__(self, channels, img_dim, activation='tanh'):
        super(CNNDecoder, self).__init__()
        self.layers = []
        for i in range(len(channels)-1):
            self.layers.append(nn.ConvTranspose2d(channels[i], channels[i+1],\
                                        kernel_size=3, stride=2, padding=1, output_padding=1))
            self.layers.append(nn.BatchNorm2d(channels[i+1]))
            if activation == 'tanh':
                    self.layers.append(nn.Tanh())
            elif activation == 'relu':
                self.layers.append(nn.ReLU())
        # Composing all convolutional layers
        width_outdim = img_dim//(2**(len(channels)-1))
        self.net = nn.Sequential(nn.Unflatten(1, 
                torch.Size([channels[0], width_outdim, width_outdim])), *self.layers)
    
    def forward(self, x):
        return self.net(x)

class VAE(nn.Module):
    """
    A VAE that can learn (either compressed or full) 1D tensor representation of original 2D tensor image (map to 3D tensor by unsqueeze 1 dimension)
    Each time image size is cut by half
    Args:
        img_dim (int): width-height dimension of 3D tensor
        latent_dim (int): dimension of latent represenation
        channels (List[int]): list of number of channels to be applied starting with original number of channels of input
        mlp_encode_dims (List[int]): densely connected layers mapping from output 1D tensor of CNN encoder to latent vector. 
        (Note this doesn't count either input (1D CNN output) or output layers (latent rep))
        mlp_decode_dims (List[int]): densely connected layers mapping from latent vector to 1D tensor input of CNN decoder
        activation_cnn (str): type of activations for convolutional layers
        activation_mlp (str): type of activations for dense layers
    """
    def __init__(self, img_dim, latent_dim, channels, mlp_encode_dims=[], mlp_decode_dims=[], activation_cnn='relu', activation_mlp='relu'):
        super(VAE, self).__init__()
        cnn_outdim = (img_dim//(2**(len(channels)-1)))**2*channels[-1]
        self.img_dim = img_dim
        self.latent_dim = latent_dim
        self.encoder = CNNEncoder(channels, activation_cnn)
        self.decoder = CNNDecoder(channels[::-1], img_dim, activation_cnn)
        self.mu_fc = Mlp(cnn_outdim, latent_dim, layer_dims=mlp_encode_dims, activation=activation_mlp)
        self.logvar_fc = Mlp(cnn_outdim, latent_dim, layer_dims=mlp_encode_dims, activation=activation_mlp)
        self.mlp_decoder = Mlp(latent_dim, cnn_outdim, layer_dims=mlp_decode_dims, activation=activation_mlp)  
        self.last_layer = nn.Sigmoid()
        
        # Store set of x and z1 for traing 
        self.x_buf = []
        self.x_recon_buf = []
        self.mu_buf = []
        self.logvar_buf = []
        
    def encode(self, x):
        z_aug = self.encoder(x.unsqueeze(1))
        mu = self.mu_fc(z_aug)
        logvar = self.logvar_fc(z_aug)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        # For now, just do Gaussian output. Consider normalizing flow such as IAF later
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        temp = self.mlp_decoder(z)
        x = self.last_layer(self.decoder(temp))
        return x.squeeze(1)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

########### Testing ###########
test = False
if test:
    # MLP
    net = Mlp(input_dim=4, output_dim=3, layer_dims=[2, 5, 6])
    inputs = torch.zeros(10, 4)
    outputs = net(inputs)
    print(outputs.shape) # Shape (10, 3)

    # encoder
    net = CNNEncoder(channels=[1, 2, 8])
    inputs = torch.zeros(10, 1, 16, 16)
    outputs = net(inputs)
    print(outputs.shape) # Shape (10, 8*4*4)
    
    # decoder
    net = CNNDecoder(channels=[8, 2, 1], img_dim = 16)
    inputs = torch.zeros(10, 8*4*4)
    outputs = net(inputs)
    print(outputs.shape) # Shape (10, 1, 16, 16)
    
    # vae
    net = VAE(channels=[1, 2, 8], img_dim = 16, latent_dim=8, mlp_encode_dims=[32, 64], mlp_decode_dims=[64, 32])
    inputs = torch.zeros(10, 16, 16)
    mu, logvar = net.encode(inputs)
    latent = net.reparameterize(mu, logvar)
    print(latent.shape)
    reconstruct = net.decode(latent)
    print(reconstruct.shape)