import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np  # Import NumPy for efficient indexing

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.hidden = nn.Linear(1, 10)
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

def train_and_get_weights(boundary):
    # Generate training data
    num_points = 1000
    X_train_below = torch.linspace(boundary - 10, boundary - 0.1, num_points // 2).unsqueeze(1)
    X_train_above = torch.linspace(boundary + 0.1, boundary + 10, num_points // 2).unsqueeze(1)
    X_train = torch.cat((X_train_below, X_train_above), dim=0)
    y_train = (X_train < boundary).float()  # 1 if number < boundary, else 0

    # Initialize the model, loss function, and optimizer
    model = MLPClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 55
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Return the weight matrices and training data as NumPy arrays
    hidden_weights = model.hidden.weight.detach().numpy()
    output_weights = model.output.weight.detach().numpy()
    X_train = X_train.detach().numpy()
    y_train = y_train.detach().numpy()

    return boundary, X_train, y_train, hidden_weights, output_weights

# Generate 1000 random boundaries and save the results
results = []
for _ in range(1000):
    boundary = random.uniform(-20, 20)
    boundary, X_train, y_train, hidden_weights, output_weights = train_and_get_weights(boundary)

    # Sample a random subset of X_train and y_train
    subset_size = 100  # Specify the number of samples you want in the subset
    indices = np.random.choice(len(X_train), size=subset_size, replace=False)

    # Index before converting to lists
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]

    # Convert to lists after indexing
    X_train_subset = X_train_subset.tolist()
    y_train_subset = y_train_subset.tolist()

    results.append({
        'boundary': boundary,
        'X_train': X_train_subset,
        'y_train': y_train_subset,
        'hidden_weights': hidden_weights.tolist(),
        'output_weights': output_weights.tolist()
    })

# Save to JSON file
with open('1data.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved to 1data.json")
