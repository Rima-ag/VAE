import argparse
import torchvision
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from vae import VAE

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--eval', type=str, default=None)
args = parser.parse_args()


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

data_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(data_train, batch_size = 64)
eval_loader = torch.utils.data.DataLoader(data_train, batch_size = 64)

def train(model, device, train_loader, opt, epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.squeeze().to(device)

        opt.zero_grad()
        pred, mu, var = model(data)
        
        L_dict = model.loss_function(data,pred, mu, var, len(train_loader.dataset))
        L_dict['L'].backward()
        opt.step()

        if batch_idx % 500 == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_reg: {:.6f}\tLoss_recon: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), (L_dict['L']).item(), L_dict['L_reg'].item(), L_dict['L_recon'].item()))

def eval(model,device, eval_loader):
    model.eval()
    pred = []
    with torch.no_grad():
      for batch_idx, (data, _) in enumerate(train_loader):
          data = data.squeeze().to(device)

          pred, _, _ = model(data)
          return pred


device = torch.device("cpu")
import torch.optim as optim
model = VAE(784, 20).to(device)

opt = optim.Adam(model.parameters(), lr=.001)

if args.eval is None:
  for epoch in range(80):
    train(model, device, train_loader, opt, epoch)
    if epoch in [9, 79]:
      torch.save(model.state_dict(), f'./models/model_{epoch}.pth')

if args.eval is not None:
  model = VAE(784, 20).to(device)
  model.load_state_dict(torch.load(args.eval), strict= False)

  pred = eval(model, device, eval_loader)
  images = torch.reshape(pred, (64, 1, 28 ,28))
  images = torchvision.transforms.Resize(512)(images)
  grid_img = torchvision.utils.make_grid(images, nrow=8, padding = 100,)

  plt.imshow(grid_img.permute(1, 2, 0))
  plt.axis('off')
  plt.show()