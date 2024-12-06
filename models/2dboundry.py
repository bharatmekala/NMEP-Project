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

def train_meta_model(num_epochs=1000, num_samples=100, batch_size=32):
    meta_model = MetaModel().to(device)
    optimizer = optim.Adam(meta_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

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
            X = np.random.uniform(-10, 10, (num_samples, 2))
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

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Feedback
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return meta_model

def evaluate_meta_model(meta_model, slope, intercept, num_test_samples=1000):
    meta_model.eval()
    with torch.no_grad():
        # Generate test data
        X_test = np.random.uniform(-10, 10, (num_test_samples, 2))
        y_test = (X_test[:, 1] <= slope * X_test[:, 0] + intercept).astype(np.float32)

        # Convert to tensor and move to device
        X_tensor = torch.from_numpy(X_test.astype(np.float32)).to(device)
        boundaries = np.array([[slope, intercept]] * num_test_samples, dtype=np.float32)
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
        print(f'Accuracy for boundary (slope={slope}, intercept={intercept}): {accuracy * 100:.2f}%')

# Example usage
if __name__ == "__main__":
    meta_model = train_meta_model()
    # Test the meta model with a specific boundary
    test_slope = 2
    test_intercept = -1.4363489579318
    evaluate_meta_model(meta_model, test_slope, test_intercept)
