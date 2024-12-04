import torch
import torch.nn as nn
import torch.optim as optim

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.hidden = nn.Linear(2, 10)
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

def train_and_get_weights(slope, intercept):
    # Generate training data
    num_points = 1000
    X_train = torch.rand(num_points, 2) * 20 - 10  # Random points in the range [-10, 10] for both dimensions
    y_train = (X_train[:, 1] < slope * X_train[:, 0] + intercept).float().view(-1, 1)  # 1 if below the line, else 0

    # Initialize the model, loss function, and optimizer
    model = MLPClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Calculate accuracy
    model.eval()
    with torch.no_grad():
        predictions = model(X_train)
        predicted_classes = (predictions > 0.5).float()
        accuracy = (predicted_classes == y_train).float().mean().item()

    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Return the weight matrices
    hidden_weights = model.hidden.weight.data
    output_weights = model.output.weight.data
    return hidden_weights, output_weights

# Example usage
slope = 1
intercept = 0
hidden_weights, output_weights = train_and_get_weights(slope, intercept)
print("Hidden layer weights:\n", hidden_weights)
print("Output layer weights:\n", output_weights)
