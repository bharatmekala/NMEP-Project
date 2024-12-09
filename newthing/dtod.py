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
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2) # mean and logvar
        )
    
    def forward(self, x):
        out = self.fc(x)
        mu, logvar = out.chunk(2, dim=1)
        return mu, logvar

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

# VAE with multiple decoders
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dims):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoders = nn.ModuleDict({
            task: Decoder(latent_dim, output_dim) for task, output_dim in output_dims.items()
        })
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, task):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoders[task](z), mu, logvar

# Define Tasks
tasks = {
    "classify_boundary": {"input_dim": 2, "output_dim": 1},
    "add_numbers": {"input_dim": 2, "output_dim": 1},
    "multiply_numbers": {"input_dim": 2, "output_dim": 1},
    "multiclass_boundary": {"input_dim": 2, "output_dim": 2}
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
            y = (x[:,0] + x[:,1] > 0).float().unsqueeze(1)
        elif task == "add_numbers":
            a = torch.randint(0, 100, (num,1)).float()
            b = torch.randint(0, 100, (num,1)).float()
            x = torch.cat([a, b], dim=1)
            y = (a + b).float()
        elif task == "multiply_numbers":
            a = torch.randint(0, 100, (num,1)).float()
            b = torch.randint(0, 100, (num,1)).float()
            x = torch.cat([a, b], dim=1)
            y = (a * b).float()
        elif task == "multiclass_boundary":
            x = torch.randn(num, 2)
            y = torch.zeros(num,2)
            y[ (x[:,0] + x[:,1] > 0), 0] = 1
            y[ (x[:,0] - x[:,1] > 0), 1] = 1
        return x, y
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Initialize Model
latent_dim = 10
output_dims = {task: info["output_dim"] for task, info in tasks.items()}
model = VAE(input_dim=2, latent_dim=latent_dim, output_dims=output_dims).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training
epochs = 100
batch_size = 64
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for _ in tqdm(range(100), desc="Tasks"):
        task = random.choice(list(tasks.keys()))
        dataset = TaskDataset(task, num_samples=1000)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, mu, logvar = model(x, task)
            if task in ["classify_boundary", "multiclass_boundary"]:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(out, y)
            else:
                loss_fn = nn.MSELoss()
                loss = loss_fn(out, y)
            # VAE loss
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss += kl / x.size(0)
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
            out, _, _ = model(x, task)
            if task == "classify_boundary":
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
            elif task == "multiclass_boundary":
                preds = torch.sigmoid(out) > 0.5
                correct += ((preds.float() == y).all(dim=1)).sum().item()
                total += y.size(0)
            else:
                preds = out
                correct += ((preds - y).abs() < 0.5).float().sum().item()
                total += y.size(0)
        accuracy = correct / total
        print(f"Task: {task}, Accuracy: {accuracy:.2f}")