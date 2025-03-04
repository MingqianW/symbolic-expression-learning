# train_simple.py
from SimpleCNN.model import SimpleCNN
from data_loader import create_dataloaders
import torch
import torch.nn as nn

def create_simple_model(device='cpu'):
    model = SimpleCNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.to(device)
    print(f"device: {device}")
    print("model info:")
    print(model)
    return model, criterion, optimizer


def train_simple_model(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    train_dataset=None,
    val_dataset=None,
    epochs = 30,
    batch_size = 32
):
    model, criterion, optimizer = create_simple_model(device)

    train_loader, val_loader, _ = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset=None, 
        batch_size=batch_size,
        shuffle_train=True
    )
    
    for epoch in range(epochs): #eppch = 30
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.unsqueeze(1).to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.unsqueeze(1).to(device)
                val_loss += criterion(model(features), targets).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train_simple_model()