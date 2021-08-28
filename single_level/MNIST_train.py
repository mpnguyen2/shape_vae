import imageio
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from net_model import AE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## HELPER FUNCTIONS ##
to_pil_image = transforms.ToPILImage()

def image_to_vid(images):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('MNIST/generated/generated_images.gif', imgs)

def save_reconstructed_images(recon_img, epoch):
    save_image(recon_img.cpu(), f"MNIST/output/output{epoch}.jpg")

def save_loss_plot(train_loss, valid_loss):
    plt.figure(figsize=(10,7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('MNIST/output/graph.jpg')
    plt.show()

def final_loss(bce_loss, mu, logvar):
    BCE=bce_loss
    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+KLD

def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        bce_loss = criterion(recon, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        
    train_loss = running_loss/counter
    
    return train_loss

def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data = data[0]
            data = data.to(device)
            recon, mu, logvar = model(data)
            bce_loss = criterion(recon, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            
            #save last batch input and then output every epoch
            if i == int(len(dataset)/dataloader.batch_size)-1:
                recon_img = recon
    
    val_loss = running_loss/counter
    return val_loss, recon_img

## Training step ##
# Set up model
model = AE().to(device)
lr = 0.001
epochs = 100
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

imgs= []

# Load data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Train set
trainset = torchvision.datasets.MNIST(
    root='MNIST/data', train=True, download=False, transform=transform
)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)

# Test set
testset = torchvision.datasets.MNIST(
    root='MNIST/data', train=False, download=False, transform=transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)

# Train loop
train_loss=[]
valid_loss=[]
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(model, trainloader, trainset, device, optimizer, criterion)
    valid_epoch_loss, recon_img = validate(model, testloader, testset, device, criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    
    save_reconstructed_images(recon_img, epoch+1)
    img = make_grid(recon_img.detach().cpu())
    imgs.append(img)
    print(f"Train loss: {train_epoch_loss:.4f}")
    print(f"Val loss: {valid_epoch_loss:.4f}")

image_to_vid(imgs)
save_loss_plot(train_loss, valid_loss)

# Save models
model_path = 'models/ae_mnist1'
torch.save(model.state_dict(), model_path) 

print('Training complete.')