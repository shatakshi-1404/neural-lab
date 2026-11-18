import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .common import train_classifier

class MNISTMLP(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

def run(epochs=2, batch_size=128):
    transform = transforms.ToTensor()
    train = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    test = datasets.MNIST("datasets", train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size)
    model = MNISTMLP()
    history = train_classifier(model, train_loader, test_loader, epochs)
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    run(parser.parse_args().epochs)
