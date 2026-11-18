import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self, noise_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28*28), nn.Tanh()
        )
    def forward(self, noise):
        return self.network(noise).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(28*28, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, images):
        return self.network(images)

def run(epochs=2, noise_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST("datasets", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    history = []

    for epoch in range(epochs):
        epoch_total = 0.0
        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch_size = len(real_images)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_images = generator(noise).detach()
            d_optimizer.zero_grad()
            d_loss = criterion(discriminator(real_images), real_labels)
            d_loss += criterion(discriminator(fake_images), fake_labels)
            d_loss.backward()
            d_optimizer.step()

            noise = torch.randn(batch_size, noise_dim, device=device)
            g_optimizer.zero_grad()
            generated = generator(noise)
            g_loss = criterion(discriminator(generated), real_labels)
            g_loss.backward()
            g_optimizer.step()
            epoch_total += d_loss.item() + g_loss.item()

        average = epoch_total / len(loader)
        history.append(average)
        print(f"Epoch {epoch+1}: combined_loss={average:.4f}")

    with torch.no_grad():
        samples = generator(torch.randn(64, noise_dim, device=device)).cpu()
        save_image(samples, "gan_samples.png", nrow=8, normalize=True, value_range=(-1, 1))
    print("Generated samples saved to gan_samples.png")
    return generator, discriminator, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    run(parser.parse_args().epochs)
