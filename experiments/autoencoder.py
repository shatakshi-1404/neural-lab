import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def run(epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.MNIST("datasets", train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        history.append(total_loss / len(dataset))
        print(f"Epoch {epoch+1}: reconstruction_loss={history[-1]:.5f}")
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    run(parser.parse_args().epochs)
