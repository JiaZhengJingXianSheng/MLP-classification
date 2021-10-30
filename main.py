import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import train
transform = transforms.Compose(
    [transforms.Resize([400, 400]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_data = ImageFolder(root="data", transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8)
val_data = ImageFolder(root="val", transform=transform)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=8)
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(400*400 * 3, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
epochs = 5

device = "cuda:0"
train.train(net, train_loader, val_loader, epochs, optimizer, loss, device)
