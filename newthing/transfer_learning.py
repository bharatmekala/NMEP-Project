import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ttow import TaskDataset, Encoder, MetaModel
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
tasks = {
    "classify_boundary": {"input_dim": 2, "output_dim": 1},
    "add_numbers": {"input_dim": 2, "output_dim": 1},
    "subtract_numbers": {"input_dim": 2, "output_dim": 1},
    "multiclass_boundary": {"input_dim": 2, "output_dim": 1}
}



latent_dim = 10
encoder = Encoder(input_dim=3, latent_dim=10).to(device)
encoder.load_state_dict(torch.load('encoder_weights.pth'))
encoder.eval()  # Set encoder to evaluation mode


# Parameters for the MLP
mlp_input_dim = 2
mlp_hidden_dim = 128  # Increased hidden dimension
mlp_output_dim = 1  # Output dimension is 1 for all tasks

# Initialize the MetaModel
latent_dim = 10  # Assuming latent_dim is defined elsewhere
meta_model = MetaModel(latent_dim, mlp_input_dim, mlp_hidden_dim, mlp_output_dim).to(device)
meta_model.load_state_dict(torch.load('meta_model_weights.pth'))
meta_model.eval()
# Assuming tasks, TaskDataset, encoder, and meta_model are defined as in ttow.py

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h = self.relu(self.fc1(x))
        out = self.fc2(h)
        return out

def train_random_mlps(task_name, num_models=1, convergence_loss=0.01, max_epochs=1000):
    input_dim = tasks[task_name]['input_dim']
    output_dim = tasks[task_name]['output_dim']
    hidden_dim = mlp_hidden_dim  # Use the same hidden dimension as in the MetaModel
    epochs_list = []
    for _ in range(num_models):
        model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        dataset = TaskDataset(task_name, num_samples=1000)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        for epoch in range(max_epochs):
            total_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            if avg_loss <= convergence_loss:
                epochs_list.append(epoch + 1)
                break
        else:
            epochs_list.append(max_epochs)
    average_epochs = sum(epochs_list) / len(epochs_list)
    print(f"Task: {task_name}, Random MLP average convergence epochs: {average_epochs}")
    return average_epochs

def train_metamodel_mlp(task_name, convergence_loss=0.01, max_epochs=1000):
    input_dim = tasks[task_name]['input_dim']
    output_dim = tasks[task_name]['output_dim']
    hidden_dim = mlp_hidden_dim
    dataset = TaskDataset(task_name, num_samples=1000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    # Generate weights using the meta_model
    xy_samples = []
    for x, y in loader:
        xy = torch.cat((x, y), dim=1)
        xy_samples.append(xy)
        break  # Need one batch for generating z
    xy_samples = torch.cat(xy_samples, dim=0).to(device)
    with torch.no_grad():
        z = encoder(xy_samples)
        # Set task indicator
        if task_name == "add_numbers":
            task_indicator = torch.ones(z.size(0), 1).to(z.device)
        elif task_name == "subtract_numbers":
            task_indicator = torch.full((z.size(0), 1), 2).to(z.device)
        else:
            task_indicator = torch.zeros(z.size(0), 1).to(z.device)
        z = torch.cat((z, task_indicator), dim=1)
        W1, W2 = meta_model(z)
        # Average the weights over the batch
        W1_mean = W1.mean(dim=0)
        W2_mean = W2.mean(dim=0)
    # Initialize an MLP with these weights
    model = SimpleMLP(input_dim, hidden_dim, output_dim).to(device)
    with torch.no_grad():
        model.fc1.weight.copy_(W1_mean)
        model.fc2.weight.copy_(W2_mean)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    for epoch in range(max_epochs):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if avg_loss <= convergence_loss:
            print(f"Task: {task_name}, MetaModel MLP converged in {epoch + 1} epochs")
            return epoch + 1
    print(f"Task: {task_name}, MetaModel MLP did not converge within {max_epochs} epochs")
    return max_epochs

# Run the experiments
for task_name in tasks.keys():
    print(f"Running experiments for task: {task_name}")
    average_random_epochs = train_random_mlps(task_name)
    metamodel_epochs = train_metamodel_mlp(task_name)
    print(f"Task: {task_name}, Random MLP average convergence epochs: {average_random_epochs}")
    print(f"Task: {task_name}, MetaModel MLP convergence epochs: {metamodel_epochs}")
