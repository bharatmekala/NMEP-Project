import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        z = self.fc(x)
        return z

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, z):
        return self.fc(z)

# AE with multiple decoders
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoders = nn.ModuleDict({
            task: Decoder(latent_dim, output_dim) for task, output_dim in output_dims.items()
        })
    
    def forward(self, x, task):
        z = self.encoder(x)
        out = self.decoders[task](z)
        return out

# Define Tasks
tasks = {
    "classify_boundary": {"input_dim": 2, "output_dim": 1},
    "add_numbers": {"input_dim": 2, "output_dim": 1},
    "subtract_numbers": {"input_dim": 2, "output_dim": 1}
    # "multiclass_boundary": {"input_dim": 2, "output_dim": 1}
}

# Dummy Dataset
class TaskDataset(Dataset):
    def __init__(self, task, num_samples=1000):
        self.task = task
        self.num_samples = num_samples
        self.data, self.labels = self.generate_data(task, num_samples)
    
    def generate_data(self, task, num):
        if task == "classify_boundary":
            x = torch.randn(num, 2)
            y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        elif task == "add_numbers":
            a = torch.randint(0, 10, (num, 1)).float()
            b = torch.randint(0, 10, (num, 1)).float()
            x = torch.cat([a, b], dim=1)
            y = a + b
        elif task == "subtract_numbers":
            a = torch.randint(0, 10, (num, 1)).float()
            b = torch.randint(0, 10, (num, 1)).float()
            x = torch.cat([a, b], dim=1)
            y = a - b
        elif task == "multiclass_boundary":
            x = torch.randn(num, 2)
            y = torch.zeros(num, 1)
            y[(x[:, 0] + x[:, 1] > 0), 0] = 1
            y[(x[:, 0] - x[:, 1] > 0), 0] += 1
        return x, y
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Initialize Model
latent_dim = 10
output_dims = {task: info["output_dim"] for task, info in tasks.items()}
model = Autoencoder(input_dim=3, latent_dim=latent_dim, output_dims=output_dims).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    # Training
    epochs = 10
    batch_size = 64
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for _ in tqdm(range(100), desc="Tasks"):
            task = random.choice(list(tasks.keys()))
            dataset = TaskDataset(task, num_samples=1000)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                xy = torch.cat((x, y), dim=1)
                optimizer.zero_grad()
                out = model(xy, task)
                if task in ["classify_boundary", "multiclass_boundary"]:
                    loss_fn = nn.BCEWithLogitsLoss()
                    loss = loss_fn(out, y)
                else:
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()

    # Testing
    model.eval()
    with torch.no_grad():
        for task in tasks.keys():
            dataset = TaskDataset(task, num_samples=200)
            loader = DataLoader(dataset, batch_size=batch_size)
            correct = 0
            total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                xy = torch.cat((x, y), dim=1)
                out = model(xy, task)
                if task == "classify_boundary" or "multiclass_boundary":
                    preds = (torch.sigmoid(out) > 0.5).float()
                    correct += (preds == y).sum().item()
                    total += y.size(0)
                else:
                    preds = out
                    correct += ((preds - y).abs() < 0.5).float().sum().item()
                    total += y.size(0)
            accuracy = correct / total
            print(f"Task: {task}, Accuracy: {accuracy:.2f}")

    # Saving the encoder's weights
    torch.save(model.encoder.state_dict(), 'encoder_weights.pth')

    # Saving each decoder's weights
    for task_name, decoder in model.decoders.items():
        torch.save(decoder.state_dict(), f'decoder_{task_name}_weights.pth')

    # Loading the encoder's weights
    model.encoder.load_state_dict(torch.load('encoder_weights.pth'))

    # Loading each decoder's weights
    for task_name, decoder in model.decoders.items():
        decoder.load_state_dict(torch.load(f'decoder_{task_name}_weights.pth'))
