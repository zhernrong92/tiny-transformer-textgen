import torch
import torch.optim as optim
import torch.nn as nn
from model import model, tokenizer

# Define Dataset
dataset = [
    "Hello, how are you?",
    "The sky is blue.",
    "I love machine learning.",
    "Transformers are powerful models."
]

# Tokenize Data
tokenized_dataset = [tokenizer(text, return_tensors="pt") for text in dataset]

# Training Setup
optimizer = optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(10):  # Train for 10 epochs
    for batch in tokenized_dataset:
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output.view(-1, tokenizer.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save trained model
torch.save(model.state_dict(), "trained_transformer.pth")
