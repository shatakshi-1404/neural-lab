import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .common import train_classifier

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def run(epochs=2, batch_size=128):
    transform = transforms.ToTensor()
    train = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    test = datasets.MNIST("datasets", train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size)
    model = MNISTCNN()
    history = train_classifier(model, train_loader, test_loader, epochs)
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    run(parser.parse_args().epochs)
