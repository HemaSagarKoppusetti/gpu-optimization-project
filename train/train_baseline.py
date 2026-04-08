import torch
import torch.nn as nn
import torch.optim as optim
import time

from models.cnn_model import SimpleCNN
from data.load_data import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():

    train_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    torch.save(model.state_dict(), "baseline_model.pth")

if __name__ == "__main__":
    train()