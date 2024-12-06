import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetaModel(nn.Module):
    def __init__(self):
        super(MetaModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Outputs W and b
        )

    def forward(self, boundary):
        out = self.net(boundary)
        W = out[:, 0].unsqueeze(1)
        b = out[:, 1].unsqueeze(1)
        return W, b

def train_meta_model(num_epochs=1000, num_samples=100, batch_size=32):
    meta_model = MetaModel().to(device)
    optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        # Sample boundaries
        boundaries = np.random.uniform(-10, 10, batch_size)
        boundaries_tensor = torch.from_numpy(boundaries.astype(np.float32)).unsqueeze(1).to(device)

        # Generate sample data for each boundary
        X_list = []
        y_list = []
        for boundary in boundaries:
            X = np.random.randint(-100, 100, num_samples)
            y = np.where(X <= boundary, 1.0, 0.0)
            X_list.append(X)
            y_list.append(y)

        X_tensor = torch.from_numpy(np.concatenate(X_list).astype(np.float32)).unsqueeze(1).to(device)
        y_tensor = torch.from_numpy(np.concatenate(y_list).astype(np.float32)).unsqueeze(1).to(device)
        boundaries_expanded = boundaries_tensor.repeat_interleave(num_samples, dim=0)

        # Get weights and biases from MetaModel
        W, b = meta_model(boundaries_expanded)

        # Predictions
        outputs = torch.sigmoid(W * X_tensor + b)

        # Compute loss
        loss = criterion(outputs, y_tensor)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Feedback
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return meta_model

def evaluate_meta_model(meta_model, test_boundary, real_boundary, num_test_samples=1000):
    meta_model.eval()
    with torch.no_grad():
        # Generate test data
        X_test = np.random.randint(-100, 100, num_test_samples)
        y_test = np.where(X_test <= real_boundary, 1.0, 0.0)

        # Convert to tensor and move to device
        X_tensor = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(1).to(device)
        boundaries = np.full(num_test_samples, test_boundary, dtype=np.float32)
        boundaries_tensor = torch.from_numpy(boundaries).unsqueeze(1).to(device)

        # Get weights and biases from MetaModel
        W, b = meta_model(boundaries_tensor)

        # Compute outputs
        outputs = torch.sigmoid(W * X_tensor + b)

        # Predictions
        preds = (outputs >= 0.5).float()

        # Move preds and y_test to CPU for comparison
        preds_cpu = preds.squeeze().cpu()
        y_test_cpu = torch.from_numpy(y_test.astype(np.float32))

        # Compute accuracy
        correct = (preds_cpu == y_test_cpu).sum().item()
        accuracy = correct / num_test_samples

        # Print accuracy
        print(f'Accuracy for boundary {real_boundary}: {accuracy * 100:.2f}%')

# Example usage
meta_model = train_meta_model()
# Test the meta model with a specific boundary
evaluate_meta_model(meta_model, predicted_boundary, boundry)

