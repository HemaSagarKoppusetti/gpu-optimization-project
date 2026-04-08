import torch
import torch.nn as nn
import torch.optim as optim

from torch.profiler import profile, record_function, ProfilerActivity

from models.cnn_model import SimpleCNN
from data.load_data import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_profiler():

    train_loader, _ = get_dataloaders(batch_size=128)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        for batch_idx, (images, labels) in enumerate(train_loader):

            if batch_idx >= 50:  # limit profiling
                break

            images, labels = images.to(device), labels.to(device)

            with record_function("forward_pass"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            with record_function("backward_pass"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print("\n=== PROFILER SUMMARY ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Save trace for visualization
    prof.export_chrome_trace("results/trace.json")

if __name__ == "__main__":
    run_profiler()