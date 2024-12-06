import json
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load and preprocess data
with gzip.open('1data_optimized.json.gz', 'rt') as f:
    data = json.load(f)

# Concatenate hidden_weights and output_weights for each entry after flattening
weight_matrices = np.array([
    np.concatenate([
        np.array(entry['hidden_weights']).flatten(),
        np.array(entry['output_weights']).flatten()
    ])
    for entry in data
])
X = torch.tensor(weight_matrices, dtype=torch.float32)

# Define Encoder with more layers
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, latent_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, latent_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)

# Define Decoder with more layers
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size2, hidden_size1, output_size):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

# Define MLPClassifier
class MLPClassifier(nn.Module):
    def __init__(self, hidden_weights, output_weights):
        super(MLPClassifier, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights with proper cloning and detaching to avoid warnings
        with torch.no_grad():
            self.hidden.weight.copy_(hidden_weights.clone().detach())
            self.hidden.bias.zero_()
            self.output.weight.copy_(output_weights.clone().detach())
            self.output.bias.zero_()
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Initialize models with more layers
input_size = X.shape[1]  # Should be 20 if hidden_weights and output_weights are both size 10
hidden_size1 = 256
hidden_size2 = 128
latent_dim = 64
output_size = input_size

encoder = Encoder(input_size, hidden_size1, hidden_size2, latent_dim)
decoder = Decoder(latent_dim, hidden_size2, hidden_size1, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Load boundary data
boundary_data = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float32)
boundary_labels = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

# Move boundary data to device if using GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder.to(device)
decoder.to(device)
boundary_data = boundary_data.to(device)
boundary_labels = boundary_labels.to(device)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    
    # Forward pass through Encoder and Decoder
    encoded = encoder(X.to(device))       # Shape: [N, latent_dim]
    decoded = decoder(encoded)            # Shape: [N, output_size]
    
    # Split decoded weights back into hidden and output weights
    hidden_size_flat = 10  # Number of hidden units in MLPClassifier
    output_size_flat = 10  # Number of output units in MLPClassifier
    
    # Ensure that decoded has enough elements
    if decoded.shape[1] < hidden_size_flat + output_size_flat:
        raise ValueError(f"Decoded output size {decoded.shape[1]} is smaller than expected {hidden_size_flat + output_size_flat}")
    
    # Use the first sample's weights and reshape
    hidden_weights = decoded[:, :hidden_size_flat]  # Shape: [N, 10]
    output_weights = decoded[:, hidden_size_flat:hidden_size_flat + output_size_flat]  # Shape: [N, 10]
    
    # Select the first set of weights for the MLPClassifier
    hidden_weights_sample = hidden_weights[0].unsqueeze(1)  # Shape: [10, 1]
    output_weights_sample = output_weights[0].unsqueeze(0)  # Shape: [1, 10]
    
    # Initialize MLP with reconstructed weights
    classifier = MLPClassifier(hidden_weights_sample, output_weights_sample).to(device)
    
    # Forward pass through MLP
    outputs_mlp = classifier(boundary_data)  # Shape: [3, 1]
    loss_cls = criterion(outputs_mlp, boundary_labels)
    
    # Backward pass and optimization for autoencoder based on MLP loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    
    # Print losses
    print(f'Epoch {epoch+1}, MLP Classification Loss: {loss_cls.item()}')

# Evaluate Autoencoder Performance
encoder.eval()
decoder.eval()
with torch.no_grad():
    encoded = encoder(X.to(device))
    decoded = decoder(encoded)
    
    hidden_weights = decoded[:, :hidden_size_flat]  # Shape: [N, 10]
    output_weights = decoded[:, hidden_size_flat:hidden_size_flat + output_size_flat]  # Shape: [N, 10]
    
    # Select the first set of weights for evaluation
    hidden_weights_sample = hidden_weights[0].unsqueeze(1)  # Shape: [10, 1]
    output_weights_sample = output_weights[0].unsqueeze(0)  # Shape: [1, 10]
    
    # Initialize MLP with reconstructed weights
    classifier = MLPClassifier(hidden_weights_sample, output_weights_sample).to(device)
    
    # Forward pass through MLP
    outputs_mlp = classifier(boundary_data)  # Shape: [3, 1]
    preds = (outputs_mlp >= 0.5).float()
    
    # Calculate accuracy
    correct = (preds == boundary_labels).sum().item()
    total = boundary_labels.size(0)
    accuracy = (correct / total) * 100
    
    # Print accuracy
    print(f'Autoencoder Test Accuracy: {correct}/{total} ({accuracy:.2f}%)')
