import torch
import torch.nn as nn
import json

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

# Define the encoder model
class Encoder(nn.Module):
    def __init__(self, task_vector_size=50):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(128, task_vector_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        task_vector = self.fc(hidden[-1])
        return task_vector

# Define the decoder model
class Decoder(nn.Module):
    def __init__(self, task_vector_size=50, weight_matrix_size=20):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(task_vector_size, 128),
            nn.ReLU(),
            nn.Linear(128, weight_matrix_size)
        )
    
    def forward(self, task_vector):
        weights = self.decoder(task_vector)
        return weights

# Load the JSON data
with open('1data.json', 'r') as f:
    data = json.load(f)

# Prepare data for training
X_data = []
y_weights = []

for item in data:
    X_train = torch.tensor(item['X_train'], dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(item['y_train'], dtype=torch.float32).view(-1, 1)
    combined = torch.cat((X_train, y_train), dim=1)  # Combine data and labels
    X_data.append(combined)
    # Flatten and concatenate hidden and output weights
    hidden_weights = torch.tensor(item['hidden_weights'], dtype=torch.float32).view(-1)
    output_weights = torch.tensor(item['output_weights'], dtype=torch.float32).view(-1)
    weights = torch.cat((hidden_weights, output_weights), dim=0)
    y_weights.append(weights)

X_data = torch.stack(X_data)  # Shape: [batch_size, num_points, 2]
y_weights = torch.stack(y_weights)  # Shape: [batch_size, weight_matrix_size]

# Initialize models
encoder = Encoder()
decoder = Decoder(weight_matrix_size=y_weights.size(1))

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    task_vectors = encoder(X_data)  # No need to flatten
    output_weights = decoder(task_vectors)
    loss = criterion(output_weights, y_weights)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Example usage
test_index = 0  # Index of the test sample
X_train = torch.tensor(data[test_index]['X_train'], dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(data[test_index]['y_train'], dtype=torch.float32).view(-1, 1)
test_combined = torch.cat((X_train, y_train), dim=1).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    task_vector = encoder(test_combined)
    predicted_weights = decoder(task_vector).squeeze(0)
    
    # Split predicted weights into hidden and output weights
    hidden_weights_size = 10 * 1  # hidden layer weights: [10, 1]
    output_weights_size = 1 * 10  # output layer weights: [1, 10]
    
    hidden_weights_flat = predicted_weights[:hidden_weights_size]
    output_weights_flat = predicted_weights[hidden_weights_size: hidden_weights_size + output_weights_size]
    
    # Reshape weights
    hidden_weights = hidden_weights_flat.view(10, 1)
    output_weights = output_weights_flat.view(1, 10)
    
    # Create a new MLP model and assign predicted weights
    new_model = MLPClassifier()
    new_model.hidden.weight.data = hidden_weights
    new_model.output.weight.data = output_weights
    
    # Evaluate the model
    new_model.eval()
    outputs = new_model(X_train)
    predicted_classes = (outputs > 0.5).float()
    accuracy = (predicted_classes == y_train).float().mean().item()
    print(f"Accuracy of the outputted model: {accuracy * 100:.2f}%")
