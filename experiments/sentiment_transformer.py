import argparse
import math
from collections import Counter
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .sentiment_lstm import SAMPLES, tokenize, SentimentDataset

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=64):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        divisors = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0)/model_dim))
        encoding = torch.zeros(max_len, model_dim)
        encoding[:, 0::2] = torch.sin(positions * divisors)
        encoding[:, 1::2] = torch.cos(positions * divisors[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, max_len=16, model_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.position = PositionalEncoding(model_dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=4, dim_feedforward=64,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(model_dim, 2)
    def forward(self, tokens):
        padding_mask = tokens == 0
        x = self.position(self.embedding(tokens))
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1)
        pooled = (encoded * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.classifier(pooled)

def run(epochs=5):
    train_rows, valid_rows = train_test_split(
        SAMPLES, test_size=0.2, random_state=42,
        stratify=[label for _, label in SAMPLES]
    )
    counts = Counter(token for text, _ in train_rows for token in tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({word: i+2 for i, (word, _) in enumerate(counts.most_common())})
    train_loader = DataLoader(SentimentDataset(train_rows, vocab), batch_size=32, shuffle=True)
    valid_loader = DataLoader(SentimentDataset(valid_rows, vocab), batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyTransformer(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(tokens), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for tokens, labels in valid_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                correct += (model(tokens).argmax(1) == labels).sum().item()
                total += len(labels)
        history["train_loss"].append(total_loss / len(train_loader.dataset))
        history["val_accuracy"].append(correct / total)
        print(f"Epoch {epoch+1}: loss={history['train_loss'][-1]:.4f}, val_acc={correct/total:.3f}")
    return model, vocab, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    run(parser.parse_args().epochs)
