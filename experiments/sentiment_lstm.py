import argparse
import re
from collections import Counter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SAMPLES = [
    ("i loved this movie it was wonderful", 1), ("great acting and a beautiful story", 1),
    ("this film was excellent and enjoyable", 1), ("what a fantastic experience", 1),
    ("the plot was boring and too slow", 0), ("i hated this terrible movie", 0),
    ("awful acting and disappointing ending", 0), ("not good and very boring", 0),
    ("the movie was surprisingly good", 1), ("a bad film with weak writing", 0),
    ("i really enjoyed the performance", 1), ("waste of time and money", 0),
    ("beautiful direction and lovely music", 1), ("the story made no sense", 0),
    ("excellent film i would watch again", 1), ("poor pacing and dull characters", 0),
] * 30

def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())

class SentimentDataset(Dataset):
    def __init__(self, rows, vocab, max_len=16):
        self.rows, self.vocab, self.max_len = rows, vocab, max_len
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        text, label = self.rows[index]
        ids = [self.vocab.get(token, 1) for token in tokenize(text)][:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(label, dtype=torch.long)

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, kind="lstm", embedding_dim=32, hidden_size=48):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        recurrent = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = recurrent(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, tokens):
        output = self.rnn(self.embedding(tokens))
        sequence = output[0] if isinstance(output, tuple) else output
        return self.fc(sequence[:, -1, :])

def run(epochs=5, kind="lstm"):
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
    model = SentimentRNN(len(vocab), kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
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
    parser.add_argument("--kind", choices=["rnn", "lstm", "gru"], default="lstm")
    args = parser.parse_args()
    run(args.epochs, args.kind)
