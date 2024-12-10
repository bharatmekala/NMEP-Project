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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, z):
        return self.fc(z)

# AE with multiple decoders
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim + 1, output_dim)
    
    def forward(self, x, task):
        z = self.encoder(x)
        # Append 1 for regression tasks, 0 for classification tasks
        if task in ["add_numbers", "subtract_numbers"]:  # Assuming these are regression tasks
            task_indicator = torch.ones(z.size(0), 1).to(z.device)
        else:
            task_indicator = torch.zeros(z.size(0), 1).to(z.device)
        z = torch.cat((z, task_indicator), dim=1)
        out = self.decoder(z)
        return out

# Define Tasks
tasks = {
    "classify_boundary": {"input_dim": 2, "output_dim": 1},
    "add_numbers": {"input_dim": 2, "output_dim": 1},
    "subtract_numbers": {"input_dim": 2, "output_dim": 1}
    #"multiclass_boundary": {"input_dim": 2, "output_dim": 1}
}

# Dummy Dataset
class TaskDataset(Dataset):
    def __init__(self, task, num_samples=1000):
        self.task = task
        self.num_samples = num_samples
        self.offset = torch.randint(0, 5, (1,)).item()  # Generate a random offset
        self.data, self.labels = self.generate_data(task, num_samples)
    
    def generate_data(self, task, num):
        if task == "classify_boundary":
            x = torch.randn(num, 2)
            y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
        elif task == "add_numbers":
            a = torch.randint(0, 10, (num, 1)).float()
            b = torch.randint(0, 10, (num, 1)).float()
            x = torch.cat([a, b], dim=1)
            y = a + b + self.offset  # Add offset to y
        elif task == "subtract_numbers":
            a = torch.randint(0, 10, (num, 1)).float()
            b = torch.randint(0, 10, (num, 1)).float()
            x = torch.cat([a, b], dim=1)
            y = a - b + self.offset  # Add offset to y
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
model = Autoencoder(input_dim=3, latent_dim=latent_dim, output_dim=1).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    # Training
    epochs = 20
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

    # Saving the encoder's weights
    torch.save(model.encoder.state_dict(), 'encoder_weights.pth')
