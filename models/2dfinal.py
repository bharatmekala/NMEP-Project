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
epochs = 100
loss_list = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

plt.figure(figsize=(5, 4.5))
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('First Phase Training Loss')
plt.savefig('first_phase_loss.png')
plt.close()

# Get predicted slope and intercept
with torch.no_grad():
    weights = model.fc.weight[0]
    bias = model.fc.bias[0]
    predicted_slope = -weights[0].item() / weights[1].item()
    predicted_intercept = -bias.item() / weights[1].item()

# Print both true and predicted values
print(f"True slope: {true_slope}, True intercept: {true_intercept}")
print(f"Predicted slope: {predicted_slope:.2f}, Predicted intercept: {predicted_intercept:.2f}")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetaModel(nn.Module):
    def __init__(self):
        super(MetaModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Outputs W1, W2, b
        )

    def forward(self, boundary):
        out = self.net(boundary)
        W1 = out[:, 0].unsqueeze(1)
        W2 = out[:, 1].unsqueeze(1)
        b = out[:, 2].unsqueeze(1)
        return W1, W2, b



def train_meta_model(num_epochs=150, num_samples=100, batch_size=32):
    meta_model = MetaModel().to(device)
    optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    loss_list = []
    
    for epoch in range(num_epochs):
        # Sample boundaries (slope and intercept)
        slopes = np.random.uniform(-5, 5, batch_size)
        intercepts = np.random.uniform(-10, 10, batch_size)
        boundaries = np.stack([slopes, intercepts], axis=1)
        boundaries_tensor = torch.from_numpy(boundaries.astype(np.float32)).to(device)

        # Generate sample data for each boundary
        X_list = []
        y_list = []
        for slope, intercept in boundaries:
            X = np.random.randint(-100, 100, (num_samples, 2))
            y = (X[:, 1] <= slope * X[:, 0] + intercept).astype(np.float32)
            X_list.append(X)
            y_list.append(y)

        X_tensor = torch.from_numpy(np.concatenate(X_list).astype(np.float32)).to(device)  # Shape: (batch_size*num_samples, 2)
        y_tensor = torch.from_numpy(np.concatenate(y_list).astype(np.float32)).unsqueeze(1).to(device)  # Shape: (batch_size*num_samples, 1)
        boundaries_expanded = boundaries_tensor.repeat_interleave(num_samples, dim=0)  # Shape: (batch_size*num_samples, 2)

        # Get weights and bias from MetaModel
        W1, W2, b = meta_model(boundaries_expanded)

        # Predictions
        outputs = torch.sigmoid(W1 * X_tensor[:, 0].unsqueeze(1) + W2 * X_tensor[:, 1].unsqueeze(1) + b)

        # Compute loss
        loss = criterion(outputs, y_tensor)
        loss_list.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Feedback
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return meta_model, loss_list

def evaluate_meta_model(meta_model, predicted_slope, predicted_intercept, real_slope, real_intercept,num_test_samples=1000):
    meta_model.eval()
    with torch.no_grad():
        # Generate test data
        X_test = np.random.uniform(-100, 100, (num_test_samples, 2))
        y_test = (X_test[:, 1] <= real_slope * X_test[:, 0] + real_intercept).astype(np.float32)

        # Convert to tensor and move to device
        X_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
        boundaries = np.array([[predicted_slope, predicted_intercept]] * num_test_samples, dtype=np.float32)
        boundaries_tensor = torch.from_numpy(boundaries).to(device)

        # Get weights and bias from MetaModel
        W1, W2, b = meta_model(boundaries_tensor)

        # Compute outputs
        outputs = torch.sigmoid(W1 * X_tensor[:, 0].unsqueeze(1) + W2 * X_tensor[:, 1].unsqueeze(1) + b)

        # Predictions
        preds = (outputs >= 0.5).float()

        # Move preds and y_test to CPU for comparison
        preds_cpu = preds.squeeze().cpu()
        y_test_cpu = torch.from_numpy(y_test.astype(np.float32))

        # Compute accuracy
        correct = (preds_cpu == y_test_cpu).sum().item()
        accuracy = correct / num_test_samples

        # Print accuracy
        print(f'Accuracy for boundary (slope={real_slope}, intercept={real_intercept}): {accuracy * 100:.2f}%')

# Example usage
if __name__ == "__main__":
    meta_model, loss_list = train_meta_model()
    plt.figure(figsize=(5, 4.5))
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Second Phase Training Loss')
    plt.savefig('second_phase_loss.png')
    plt.close()
    # Test the meta model with a specific boundary
    evaluate_meta_model(meta_model, predicted_slope, predicted_intercept, true_slope, true_intercept)
