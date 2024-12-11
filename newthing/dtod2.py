# FILE: dtod2.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, latent_dim, num_points_per_sample, input_dim_per_point, output_dim_per_point):
        super(Decoder, self).__init__()
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.hidden_to_x = nn.Sequential(
            nn.Linear(256, num_points_per_sample * input_dim_per_point)
        )
        self.hidden_to_y = nn.Sequential(
            nn.Linear(256, num_points_per_sample * output_dim_per_point)
        )
        self.num_points_per_sample = num_points_per_sample
        self.input_dim_per_point = input_dim_per_point
        self.output_dim_per_point = output_dim_per_point
    
    def forward(self, z):
        hidden = self.latent_to_hidden(z)
        x_out = self.hidden_to_x(hidden)
        y_out = self.hidden_to_y(hidden)
        # Reshape outputs to match the original data shape
        x_out = x_out.view(-1, self.num_points_per_sample, self.input_dim_per_point)
        y_out = y_out.view(-1, self.num_points_per_sample, self.output_dim_per_point)
        return x_out, y_out

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_points_per_sample, input_dim_per_point, output_dim_per_point):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, num_points_per_sample, input_dim_per_point, output_dim_per_point)
    
    def forward(self, x, task):
        z = self.encoder(x)
        # Append task indicator
        if task in ["add_numbers", "subtract_numbers"]:
            task_indicator = torch.ones(z.size(0), 1).to(z.device)
        else:
            task_indicator = torch.zeros(z.size(0), 1).to(z.device)
        z = torch.cat((z, task_indicator), dim=1)
        x_out, y_out = self.decoder(z)
        return x_out, y_out

# Define Tasks
tasks = {
    "classify_boundary": {"input_dim": 2, "output_dim": 1}
    #"add_numbers": {"input_dim": 2, "output_dim": 1},
    #"subtract_numbers": {"input_dim": 2, "output_dim": 1},
    #"multiclass_boundary": {"input_dim": 2, "output_dim": 1}
}

# Dataset
class TaskDataset(Dataset):
    def __init__(self, task, num_samples=1000, num_points_per_sample=100):
        self.task = task
        self.num_samples = num_samples
        self.num_points_per_sample = num_points_per_sample
        self.offset = torch.randint(0, 5, (1,)).item()
        self.data, self.labels = self.generate_data(task, num_samples, num_points_per_sample)
    
    def generate_data(self, task, num_samples, num_points_per_sample):
        data = []
        labels = []
        for _ in range(num_samples):
            if task == "classify_boundary":
                x = torch.randn(num_points_per_sample, 2)
                y = (x[:, 0] + x[:, 1] > 0).float().unsqueeze(1)
            elif task == "add_numbers":
                a = torch.randint(0, 10, (num_points_per_sample, 1)).float()
                b = torch.randint(0, 10, (num_points_per_sample, 1)).float()
                x = torch.cat([a, b], dim=1)
                y = a + b + self.offset
            elif task == "subtract_numbers":
                a = torch.randint(0, 10, (num_points_per_sample, 1)).float()
                b = torch.randint(0, 10, (num_points_per_sample, 1)).float()
                x = torch.cat([a, b], dim=1)
                y = a - b + self.offset
            elif task == "multiclass_boundary":
                x = torch.randn(num_points_per_sample, 2)
                y = torch.zeros(num_points_per_sample, 1)
                y[(x[:, 0] + x[:, 1] > 0), 0] = 1
                y[(x[:, 0] - x[:, 1] > 0), 0] += 1
            data.append(x)
            labels.append(y)
        return data, labels
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        # Flatten x and y
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        xy = torch.cat((x_flat, y_flat), dim=0)
        return xy

# Initialize Model
num_points_per_sample = 100
input_dim_per_point = 2
output_dim_per_point = 1
input_dim = num_points_per_sample * (input_dim_per_point + output_dim_per_point)
latent_dim = 20

model = Autoencoder(
    input_dim=input_dim,
    latent_dim=latent_dim,
    num_points_per_sample=num_points_per_sample,
    input_dim_per_point=input_dim_per_point,
    output_dim_per_point=output_dim_per_point
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    # Training and Testing
    epochs = 200
    batch_size = 64
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        for _ in tqdm(range(100), desc="Tasks"):
            task = random.choice(list(tasks.keys()))
            dataset = TaskDataset(task, num_samples=1000, num_points_per_sample=num_points_per_sample)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for xy in loader:
                xy = xy.to(device)
                optimizer.zero_grad()
                # Extract inputs and labels from xy
                total_input_dim = num_points_per_sample * input_dim_per_point
                x = xy[:, :total_input_dim]
                y = xy[:, total_input_dim:]
                # Forward pass
                x_out, y_out = model(xy, task)
                # Reshape x and y
                x = x.view(-1, num_points_per_sample, input_dim_per_point)
                y = y.view(-1, num_points_per_sample, output_dim_per_point)
                # Compute loss
                if task in ["classify_boundary", "multiclass_boundary"]:
                    loss_fn = nn.BCEWithLogitsLoss()
                    loss = loss_fn(y_out, y)
                else:
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(y_out, y)
                loss.backward()
                optimizer.step()
        
        # Testing after each epoch
        model.eval()
        per_task_correct = {task: 0 for task in tasks.keys()}
        per_task_total = {task: 0 for task in tasks.keys()}
        with torch.no_grad():
            for task in tasks.keys():
                dataset = TaskDataset(task, num_samples=200, num_points_per_sample=num_points_per_sample)
                loader = DataLoader(dataset, batch_size=batch_size)
                for xy in loader:
                    xy = xy.to(device)
                    # Extract inputs and labels from xy
                    total_input_dim = num_points_per_sample * input_dim_per_point
                    x = xy[:, :total_input_dim]
                    y = xy[:, total_input_dim:]
                    # Forward pass
                    x_out, y_out = model(xy, task)
                    # Reshape x_out and y_correct
                    x_out = x_out.view(-1, num_points_per_sample, input_dim_per_point)
                    y_correct = y.view(-1, num_points_per_sample)
                    
                    # Compute y_correct based on x_out using task rules
                    if task == "classify_boundary":
                        y_correct = (x_out[:, :, 0] + x_out[:, :, 1] > 0).float().unsqueeze(2)  # Shape: [batch, 100, 1]
                        predictions = torch.sigmoid(y_out).round()  # Shape: [batch, 100, 1]
                        per_task_correct[task] += (predictions == y_correct).sum().item()
                        per_task_total[task] += y_correct.numel()
                    
                    elif task == "add_numbers":
                        y_correct = (x_out[:, :, 0] + x_out[:, :, 1] + dataset.offset).unsqueeze(2)  # Shape: [batch, 100, 1]
                        predictions = y_out  # Shape: [batch, 100, 1]
                        # Define a tolerance for regression accuracy
                        tolerance = 0.5
                        per_task_correct[task] += ((predictions - y_correct).abs() <= tolerance).sum().item()
                        per_task_total[task] += y_correct.numel()
                    
                    elif task == "subtract_numbers":
                        y_correct = (x_out[:, :, 0] - x_out[:, :, 1] + dataset.offset).unsqueeze(2)  # Shape: [batch, 100, 1]
                        predictions = y_out  # Shape: [batch, 100, 1]
                        tolerance = 0.5
                        per_task_correct[task] += ((predictions - y_correct).abs() <= tolerance).sum().item()
                        per_task_total[task] += y_correct.numel()
                    
                    elif task == "multiclass_boundary":
                        y_correct = torch.zeros_like(y_out).to(device)  # Shape: [batch, 100, 1]
                        mask1 = (x_out[:, :, 0] + x_out[:, :, 1] > 0)
                        mask2 = (x_out[:, :, 0] - x_out[:, :, 1] > 0)
                        y_correct[mask1] = 1
                        y_correct[mask2] += 1
                        predictions = torch.sigmoid(y_out).round()  # Shape: [batch, 100, 1]
                        per_task_correct[task] += (predictions == y_correct).sum().item()
                        per_task_total[task] += y_correct.numel()
        
        # Calculate and print accuracies for each task
        print(f"\nEpoch {epoch+1} Accuracy per Task:")
        for task in tasks.keys():
            if per_task_total[task] > 0:
                accuracy = per_task_correct[task] / per_task_total[task]
                print(f"Task: {task}, Accuracy: {accuracy:.2f}")
            else:
                print(f"Task: {task}, No samples to evaluate.")

    # Save encoder weights
    torch.save(model.encoder.state_dict(), 'encoder_weights.pth')
