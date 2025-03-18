import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, num_heads, num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc(x)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")


# Initialize Model
vocab_size = tokenizer.vocab_size
model = TinyTransformer(vocab_size)

# Save model structure (for reloading)
torch.save(model.state_dict(), "tiny_transformer.pth")
