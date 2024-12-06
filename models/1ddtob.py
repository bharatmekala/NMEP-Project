import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
torch.manual_seed(0)

# Create dataset
boundry = 30.0
X = torch.randint(-100, 100, (1000, 1), dtype=torch.float32).to(device)
y = (X < boundry).float().to(device)

# Define the MLP model with 2 hidden layers
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)  # Linear output
        return x

model = BinaryClassifier().to(device)

# Define MLE Loss
def mle_loss(predicted, target):
    distribution = torch.distributions.Bernoulli(logits=predicted)
    return -distribution.log_prob(target).mean()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = mle_loss(predictions, y)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Get predicted boundary
with torch.no_grad():
    x_vals = torch.linspace(-100, 100, 1000).unsqueeze(1).to(device)
    preds = model(x_vals)
    boundary_idx = (torch.abs(preds - 0.5)).argmin()
    predicted_boundary = x_vals[boundary_idx].item()

# Print both boundaries
print(f"Real boundary: x = {boundry}")
print(f"Predicted boundary: x = {predicted_boundary:.2f}")
