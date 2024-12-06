# 1dmodel.py

import torch
import torch.nn as nn
import torch.optim as optim
import json
import gzip
import numpy as np
from tqdm import tqdm

# Define Encoder and Decoder
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, latent_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load dataset
with gzip.open('1data_optimized.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)

# Prepare data
weight_vectors = []
boundaries = []
for item in data:
    hidden_weights = np.array(item['hidden_weights']).flatten()
    output_weights = np.array(item['output_weights']).flatten()
    concatenated = np.concatenate((hidden_weights, output_weights))
    weight_vectors.append(concatenated)
    boundaries.append(item['boundary'])

weight_vectors = np.array(weight_vectors, dtype=np.float32)
boundaries = np.array(boundaries, dtype=np.float32)

# Convert to tensors
inputs = torch.from_numpy(weight_vectors)
targets = torch.from_numpy(weight_vectors)

# Define model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = inputs.shape[1]
latent_dim = 16
output_dim = input_dim

# Initialize models
encoder = Encoder(input_dim, latent_dim).to(device)
decoder = Decoder(latent_dim, output_dim).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop
num_epochs = 1
batch_size = 32
dataset = torch.utils.data.TensorDataset(inputs, targets)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    total_loss = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        latent = encoder(batch_x)
        outputs = decoder(latent)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(loader):.4f}')

# Evaluation
encoder.eval()
decoder.eval()

accuracies = []
for i, item in enumerate(data):
    original_weights = weight_vectors[i]
    boundary = boundaries[i]
    
    with torch.no_grad():
        latent = encoder(inputs[i].unsqueeze(0).to(device))
        reconstructed = decoder(latent).squeeze(0).cpu().numpy()
    
    hidden_size = 10
    hidden_weights_recon = reconstructed[:hidden_size].reshape(10,1)
    output_weights_recon = reconstructed[hidden_size:].reshape(1,10)
    
    # Define MLP with reconstructed weights
    class ReconstructedMLP(nn.Module):
        def __init__(self, hidden_w, output_w):
            super(ReconstructedMLP, self).__init__()
            self.hidden = nn.Linear(1, 10)
            self.output = nn.Linear(10, 1)
            self.hidden.weight = nn.Parameter(torch.from_numpy(hidden_w).to(device))
            self.output.weight = nn.Parameter(torch.from_numpy(output_w).to(device))
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = torch.relu(self.hidden(x))
            x = self.sigmoid(self.output(x))
            return x
    
    model = ReconstructedMLP(hidden_weights_recon, output_weights_recon).to(device)
    
    # Generate fake data
    num_points = 1000
    X_below = np.linspace(boundary - 10, boundary - 0.1, num_points // 2).reshape(-1,1)
    X_above = np.linspace(boundary + 0.1, boundary + 10, num_points // 2).reshape(-1,1)
    X_fake = np.vstack((X_below, X_above)).astype(np.float32)
    y_fake = (X_fake < boundary).astype(np.float32)
    
    X_tensor = torch.from_numpy(X_fake).to(device)
    y_tensor = torch.from_numpy(y_fake).to(device)
    
    # Evaluate accuracy
        # Evaluate accuracy
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        predictions = (outputs.cpu().numpy() >= 0.5).astype(np.float32)
        accuracy = (predictions == y_fake).mean()
        accuracies.append(accuracy)

print(f'Average MLP Accuracy: {np.mean(accuracies):.2f}')
