import torch.nn as nn
import torch 
import numpy as np
import logging
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from net_model import VAE

# Set up logger
logger = logging.getLogger("Model logger")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('out.log', mode='w')
logger.addHandler(fh)

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
    logger.debug("Train Loss: {}".format(train_loss))
    return train_loss

# Loss function for VAE
def loss_func (recon_x, x, mu, log_var):
    beta = 0.001
    binary_cross_entropy = F.binary_cross_entropy(recon_x, x, reduction="mean")
    KL_loss = beta * torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
    #KL_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return KL_loss + binary_cross_entropy

# Test MNIST dataset
# Calculates the test loss and the pixelwise MSE on the test set
def test_mnist(vae, test_loader):
    vae.eval()
    test_loss = 0
    mse_loss = 0
    with torch.no_grad():
        for image, _ in test_loader:
            # Encode, Sample, Decode 
            recon_image, mu, log_var = vae(image)
            # Calculate losses
            test_loss += loss_func(recon_image, image, mu, log_var).item()
            mse_loss += F.mse_loss(recon_image, image, reduction="sum").item()
    logger.debug('Average Test Loss: {}'.format(test_loss/len(test_loader.dataset)))
    logger.debug('MSE Loss: {}'.format(mse_loss/len(test_loader.dataset)))

# Visualizes the reconstructions and outputs the result to a PNG file
# There 5 sample reconstructions per PNG file
def visualize_recon(dataset, vae, num_samples, output="./output/visual"):
    vae.eval()
    sample_indices = np.random.choice(range(1, 10000), num_samples, replace=False)
    with torch.no_grad():
        output_num = 1
        i = 1
        fig = plt.figure()
        for sample_index in sample_indices:
            # Generate two subplots
            ax1 = fig.add_subplot(5, 2, i)
            ax2 =  fig.add_subplot(5, 2, i+1)
            # Title columns
            if i == 1:
                ax1.title.set_text('Original')
                ax2.title.set_text("Reconstructed")
            # Grab sample image and find reconstruction
            image, label = dataset[sample_index]
            image = torch.reshape(image, (1, 1, 32, 32))
            recon_image, _, _ = vae(image)
            # Place reconstructed image on the right and original images on left
            ax1.imshow(torch.reshape(image, (32, 32)), cmap='gray')
            ax2.imshow(torch.reshape(recon_image, (32, 32)), cmap='gray')
            i += 2
            # Write to output every 5 images 
            if (i + 1) % 5 == 0:
                plt.savefig(output + "_" + str(output_num))
                output_num += 1
                i = 1
                fig.clear()
        plt.close(fig)
    return 

# Dataloader for MNIST Digit Dataset
# All images are converted to tensors and normalized by the mean and standard deviation of MNIST

# Loads in the training set for MNIST
train_dataset = torchvision.datasets.MNIST('./mnist_data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((32, 32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)) ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Loads in the test set for mnist
test_dataset = torchvision.datasets.MNIST('./mnist_data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((32, 32)),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,)) ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)


vae = VAE()
# Train VAE
for epoch in range(1, 51):
    logger.debug("Epoch: {}".format(epoch))
    train_mnist_epoch(vae, train_loader)
    test_mnist(vae, test_loader)
    visualize_recon(test_dataset, vae, 10, output="./output/epoch_" + str(epoch))
    if epoch % 10:
        torch.save(vae.state_dict(), "./output/model")
    torch.save(vae.state_dict(), "./output/model")
# Reconstruct a bunch of images
with torch.no_grad():
    z = torch.randn(64, 8)
    sample = vae.decode(z)
    torchvision.utils.save_image(sample.view(64, 1, 32, 32), "./output/recon_" + ".png")


                        


