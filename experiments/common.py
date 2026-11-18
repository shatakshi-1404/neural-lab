import torch

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_classifier(model, train_loader, test_loader, epochs=2, learning_rate=1e-3):
    device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)

        history["train_loss"].append(total_loss / len(train_loader.dataset))
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        history["test_accuracy"].append(correct / max(total, 1))
        print(f"Epoch {epoch+1}/{epochs} | loss={history['train_loss'][-1]:.4f} | accuracy={history['test_accuracy'][-1]:.4f}")
    return history
