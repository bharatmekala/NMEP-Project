import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
torch.manual_seed(0)

# Function to generate fake data
def generate_data(slope, intercept, num_samples=1000):
    X = torch.randint(-100, 100, (num_samples, 2), dtype=torch.float32).to(device)
    y = (X[:, 1] > slope * X[:, 0] + intercept).float().to(device)
    return X, y

# Generate fake data
true_slope = 10.0
true_intercept = -5.0
X, y = generate_data(true_slope, true_intercept)

# Define the model
class LinearSeparator(nn.Module):
    def __init__(self):
        super(LinearSeparator, self).__init__()
        self.fc = nn.Linear(2, 1)
    
    def forward(self, x):
        return self.fc(x)

model = LinearSeparator().to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Get predicted slope and intercept
with torch.no_grad():
    weights = model.fc.weight[0]
    bias = model.fc.bias[0]
    predicted_slope = -weights[0].item() / weights[1].item()
    predicted_intercept = -bias.item() / weights[1].item()

# Print both true and predicted values
print(f"True slope: {true_slope}, True intercept: {true_intercept}")
print(f"Predicted slope: {predicted_slope:.2f}, Predicted intercept: {predicted_intercept:.2f}")

