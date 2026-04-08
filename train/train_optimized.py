import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch.cuda.amp import GradScaler, autocast

from models.cnn_model import SimpleCNN
from data.load_data import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():

    # Optimized DataLoader
    train_loader, _ = get_dataloaders(batch_size=256)

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scaler = GradScaler()

    epochs = 5

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed Precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"[Optimized] Epoch {epoch+1}, Loss: {running_loss:.4f}")

    total_time = time.time() - start_time
    print(f"[Optimized] Total Training Time: {total_time:.2f} seconds")

    torch.save(model.state_dict(), "optimized_model.pth")

if __name__ == "__main__":
    train()