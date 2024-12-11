import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Define Tasks
tasks = {
    "classify_boundary": {"input_dim": 2, "output_dim": 1},
    "add_numbers": {"input_dim": 2, "output_dim": 1},
    "subtract_numbers": {"input_dim": 2, "output_dim": 1}
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
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dtod2 import Encoder
latent_dim = 10
encoder = Encoder(input_dim=3, latent_dim=10).to(device)
encoder.load_state_dict(torch.load('encoder_weights.pth'))
encoder.eval()  # Set encoder to evaluation mode

import torch
import torch.nn as nn
import torch.optim as optim

# Define the MetaModel that generates weight matrices
class MetaModel(nn.Module):
    def __init__(self, latent_dim, mlp_input_dim, mlp_hidden_dim, mlp_output_dim):
        super(MetaModel, self).__init__()
        total_weights = (mlp_input_dim * mlp_hidden_dim) + (mlp_hidden_dim * mlp_output_dim)
        hidden_size = 1024  # Increased hidden size for the MetaModel

        # Adding more layers and dropout to the MetaModel
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, total_weights)
        )

        self.mlp_input_dim = mlp_input_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_output_dim = mlp_output_dim

    def forward(self, z):
        weights = self.fc(z)
        # Split weights into two matrices for a two-layer MLP
        w1_size = self.mlp_input_dim * self.mlp_hidden_dim
        w2_size = self.mlp_hidden_dim * self.mlp_output_dim
        W1 = weights[:, :w1_size].view(-1, self.mlp_hidden_dim, self.mlp_input_dim)
        W2 = weights[:, w1_size:].view(-1, self.mlp_output_dim, self.mlp_hidden_dim)
        return W1, W2

# Parameters for the MLP
mlp_input_dim = 2
mlp_hidden_dim = 128  # Increased hidden dimension
mlp_output_dim = 1  # Output dimension is 1 for all tasks

# Initialize the MetaModel
latent_dim = 10  # Assuming latent_dim is defined elsewhere
meta_model = MetaModel(latent_dim, mlp_input_dim, mlp_hidden_dim, mlp_output_dim).to(device)
optimizer = optim.Adam(meta_model.parameters(), lr=1e-3)

# Training loop
epochs = 10000
batch_size = 128

if __name__ == "__main__":
    # Testing phase
    def printloss():
        meta_model.eval()
        with torch.no_grad():
            for task_name, task_info in tasks.items():
                dataset = TaskDataset(task_name, num_samples=1000)
                loader = DataLoader(dataset, batch_size=batch_size)
                correct = 0
                total = 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    xy = torch.cat((x, y), dim=1)  # Concatenate along the feature dimension
                
                    with torch.no_grad():
                        z = encoder(xy)  # Feed concatenated input into the encoder
                        # Set task indicator
                        if task_name == "add_numbers":
                            task_indicator = torch.ones(z.size(0), 1).to(z.device)
                        elif task_name == "subtract_numbers":
                            task_indicator = torch.full((z.size(0), 1), 2).to(z.device)
                        else:
                            task_indicator = torch.zeros(z.size(0), 1).to(z.device)
                        z = torch.cat((z, task_indicator), dim=1)
                    
                    W1, W2 = meta_model(z)
                    x_expanded = x.unsqueeze(1)
                    h = torch.bmm(x_expanded, W1.transpose(1, 2)).squeeze(1)
                    h = torch.relu(h)
                    h_expanded = h.unsqueeze(1)
                    outputs = torch.bmm(h_expanded, W2.transpose(1, 2)).squeeze(1)
                    if task_name == "classify_boundary":
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        correct += (preds == y).sum().item()
                        total += y.size(0)
                    else:
                        preds = outputs
                        correct += ((preds - y).abs() < 0.5).float().sum().item()
                        total += y.size(0)
                accuracy = correct / total
                print(f"Task: {task_name}, Accuracy: {accuracy:.2f}")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for task_name, task_info in tasks.items():
            dataset = TaskDataset(task_name, num_samples=2000)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                xy = torch.cat((x, y), dim=1)  # Concatenate along the feature dimension
                
                with torch.no_grad():
                    z = encoder(xy)  # Feed concatenated input into the encoder
                    # Set task indicator
                    if task_name == "add_numbers":
                        task_indicator = torch.ones(z.size(0), 1).to(z.device)
                    elif task_name == "subtract_numbers":
                        task_indicator = torch.full((z.size(0), 1), 2).to(z.device)
                    else:
                        task_indicator = torch.zeros(z.size(0), 1).to(z.device)
                    z = torch.cat((z, task_indicator), dim=1)
                
                W1, W2 = meta_model(z)  # Generate weight matrices

                # Forward pass using the generated weights
                x_expanded = x.unsqueeze(1)  # Shape: [batch_size, 1, input_dim]
                h = torch.bmm(x_expanded, W1.transpose(1, 2)).squeeze(1)  # Hidden layer
                h = torch.relu(h)
                h_expanded = h.unsqueeze(1)  # Shape: [batch_size, 1, hidden_dim]
                outputs = torch.bmm(h_expanded, W2.transpose(1, 2)).squeeze(1)  # Output layer

                # Compute loss
                loss_fn = nn.MSELoss()
                loss = loss_fn(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if epoch % 100 == 0:
            printloss()
            torch.save(meta_model.state_dict(), 'meta_model_weights.pth')



