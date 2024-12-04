import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import gzip
from tqdm import tqdm
import os

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

# Define the Encoder and Decoder classes
class Encoder(nn.Module):
    def __init__(self, task_vector_size=50):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)  # Reduced hidden_size
        self.fc = nn.Linear(64, task_vector_size)
        self.dropout = nn.Dropout(p=0.5)  # Added Dropout

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        task_vector = self.fc(hidden[-1])
        task_vector = self.dropout(task_vector)
        return task_vector
# Define the decoder model
class Decoder(nn.Module):
    def __init__(self, task_vector_size=50, weight_matrix_size=20):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(task_vector_size, 64),  # Reduced layer size
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Added Dropout
            nn.Linear(64, weight_matrix_size)
        )

    def forward(self, task_vector):
        out = self.decoder(task_vector)
        return out

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, json_file, compressed=False):
        """
        Args:
            json_file (str): Path to the JSON file containing the data.
            compressed (bool): If True, the JSON file is gzip-compressed.
        """
        if compressed:
            with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process X_train and y_train
        X_train = torch.tensor(item['X_train'], dtype=torch.float32).view(-1, 1)  # [num_points, 1]
        y_train = torch.tensor(item['y_train'], dtype=torch.float32).view(-1, 1)  # [num_points, 1]
        combined = torch.cat((X_train, y_train), dim=1)  # [num_points, 2]
        
        # Process weights
        hidden_weights = torch.tensor(item['hidden_weights'], dtype=torch.float32).view(-1)  # [hidden_weights_size]
        output_weights = torch.tensor(item['output_weights'], dtype=torch.float32).view(-1)  # [output_weights_size]
        weights = torch.cat((hidden_weights, output_weights), dim=0)  # [weight_matrix_size]
        
        return combined, weights

def calculate_accuracy(encoder, decoder, data_loader, device):
    encoder.eval()
    decoder.eval()
    total_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass through encoder and decoder
            task_vector = encoder(batch_X)
            predicted_weights = decoder(task_vector)
            
            # Split predicted weights into hidden and output weights
            hidden_weights_size = 10 * 1  # hidden layer weights: [10, 1]
            output_weights_size = 1 * 10  # output layer weights: [1, 10]
            
            hidden_weights_flat = predicted_weights[:, :hidden_weights_size]
            output_weights_flat = predicted_weights[:, hidden_weights_size: hidden_weights_size + output_weights_size]
            
            # Reshape weights
            hidden_weights = hidden_weights_flat.view(-1, 10, 1)
            output_weights = output_weights_flat.view(-1, 1, 10)
            
            # Create a new MLP model and assign predicted weights
            new_model = MLPClassifier().to(device)
            new_model.hidden.weight.data = hidden_weights[0]  # Assuming batch size of 1
            new_model.output.weight.data = output_weights[0]  # Assuming batch size of 1
            
            # Evaluate the model
            new_model.eval()
            X_train = batch_X[:, :, 0].view(-1, 1)  # Use only the X_train part and reshape
            y_train = batch_X[:, :, 1].view(-1, 1)  # Use only the y_train part and reshape
            outputs = new_model(X_train)
            predicted_classes = (outputs > 0.5).float()
            accuracy = (predicted_classes == y_train).float().mean().item()
            
            total_accuracy += accuracy * batch_X.size(0)
            total_samples += batch_X.size(0)
    
    overall_accuracy = total_accuracy / total_samples
    return overall_accuracy * 100

def main():
    # Configuration
    json_path = '1data_optimized.json.gz'  # Change to '1data.json' if not compressed
    compressed_json = True  # Set to False if using uncompressed JSON
    batch_size = 1000
    num_epochs = 1000
    learning_rate = 0.02
    validation_split = 0.2
    num_workers = 4  # Adjust based on your CPU cores
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Dataset
    dataset = CustomDataset(json_file=json_path, compressed=compressed_json)
    
    # Split into training and validation sets
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize models
    sample_item = dataset[0]
    weight_matrix_size = sample_item[1].shape[0]
    encoder = Encoder().to(device)
    decoder = Decoder(weight_matrix_size=weight_matrix_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=learning_rate, 
        weight_decay=1e-4  # Added weight decay
    )    
    # Directory for checkpoints
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training Loop with Validation and Checkpointing
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()
        train_loss = 0.0
        
        # Training Phase
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training", leave=False):
            batch_X = batch_X.to(device)  # [batch_size, num_points, 2]
            batch_y = batch_y.to(device)  # [batch_size, weight_matrix_size]
            
            # Forward pass
            encoded = encoder(batch_X)  # [batch_size, 32]
            decoded = decoder(encoded)  # [batch_size, weight_matrix_size]
            
            # Compute loss
            loss = criterion(decoded, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * batch_X.size(0)
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_size
        
        # Validation Phase
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Validation", leave=False):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Forward pass
                encoded = encoder(batch_X)
                decoded = decoder(encoded)
                
                # Compute loss
                loss = criterion(decoded, batch_y)
                
                # Accumulate loss
                val_loss += loss.item() * batch_X.size(0)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_size
        val_accuracy = calculate_accuracy(encoder, decoder, val_loader, device)

        
        # Print epoch summary
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")        
        
        # Save checkpoints every 50 epochs and the final epoch
        if epoch % 50 == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
    
    # Save the final trained models
    torch.save(encoder.state_dict(), 'encoder_final.pth')
    torch.save(decoder.state_dict(), 'decoder_final.pth')
    print("Training completed and final models saved.")

if __name__ == "__main__":
    main()
