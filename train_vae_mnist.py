import torch.nn as nn
import torch 
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from net_model import VAE


# Dataloader for MNIST Digit Dataset
# All images are converted to tensors and normalized by the mean and standard deviation of MNIST

# Loads in the training set for MNIST
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./mnist_data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((32, 32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=64, shuffle=True)

# Loads in the test set for mnist
test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('./mnist_data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((32, 32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=1000, shuffle=True)


# Displays the first MNIST digit
def show_mnist(data_loader, output_path):
    data = enumerate(data_loader)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    for batch_idx, img in data:
        ax1.imshow(torch.reshape(img[0], (32, 32)), cmap='gray')
        plt.savefig(output_path)
        break

# Train mnist for one epoch
def train_mnist_epoch(vae, train_loader):
    optimizer = optim.Adam(vae.parameters())
    vae.train()
    train_loss = 0
    for batch, (image, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Encode, Sample, Decode 
        mu, log_var = vae.encode(image)
        z = vae.reparameterize(mu, log_var)
        pred_image = vae.decode(z)
        # Calc Loss and backprop
        loss = loss_func(pred_image, image, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print("Train Loss:", train_loss)
    return train_loss

# Loss function for VAE
def loss_func (recon_x, x, mu, log_var):
    binary_cross_entropy = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KL_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KL_loss + binary_cross_entropy

# Test MNIST dataset
def test_mnist(vae, test_loader):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for image, _ in test_loader:
            # Encode, Sample, Decode 
            mu, log_var = vae.encode(image)
            z = vae.reparameterize(mu, log_var)
            pred_image = vae.decode(z)
            # Calc Loss and backprop
            loss = loss_func(pred_image, image, mu, log_var)
            test_loss += loss.item()
    print('Test Loss: {}'.format(test_loss))


vae = VAE()
for epoch in range(1, 51):
    print("Epoch:", epoch)
    train_mnist_epoch(vae, train_loader)
    test_mnist(vae, test_loader)


                        


