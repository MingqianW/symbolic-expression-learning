import torch
import torch.nn as nn
from model import CustomNN
from data_loader import create_dataloaders


def create_model(n_groups=3900, hidden_dim=64, device='cpu'):
    model = CustomNN(n_groups, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    print(f"device: {device}")
    print("model info:")
    print(model)
    return model, criterion, optimizer


def train_model(
    n_groups=3,
    hidden_dim=64,
    epochs=30,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
): 
    
    train_loader, val_loader, _ = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset=None, 
        batch_size=32,
        shuffle_train=True
    )

    
    num_samples = 1000
    X_data = torch.randn(num_samples, n_groups*3)
    Y_data = torch.randn(num_samples, n_groups)
    dataset = TensorDataset(X_data, Y_data)
    
    train_loader, val_loader, _ = create_dataloaders(dataset, batch_size)
    model, criterion, optimizer = create_model(n_groups, hidden_dim, device)
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        'n_groups': 3,
        'hidden_dim': 64,
        'epochs': 50,
        'batch_size': 32
    }
    
    train_model(**CONFIG)