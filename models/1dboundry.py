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
            X = np.random.uniform(boundary - 5, boundary + 5, num_samples)
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

def evaluate_meta_model(meta_model, boundary, num_test_samples=1000):
    meta_model.eval()
    with torch.no_grad():
        # Generate test data
        X_test = np.random.uniform(boundary - 5, boundary + 5, num_test_samples)
        y_test = np.where(X_test <= boundary, 1.0, 0.0)

        # Convert to tensor and move to device
        X_tensor = torch.from_numpy(X_test.astype(np.float32)).unsqueeze(1).to(device)
        boundaries = np.full(num_test_samples, boundary, dtype=np.float32)
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
        print(f'Accuracy for boundary {boundary}: {accuracy * 100:.2f}%')

# Example usage
if __name__ == "__main__":
    meta_model = train_meta_model()
    # Test the meta model with a specific boundary
    test_boundary = -6.34613
    evaluate_meta_model(meta_model, test_boundary)
