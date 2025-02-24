# train.py
import torch
import torch.nn as nn
from model import PhysicsCNN, EnhancedPhysicsCNN
from data_loader import create_dataloaders
from data_generator import DataGenerator
import torch.optim as optim
import matplotlib.pyplot as plt

def train():
    # 1. 初始化模型
    model = EnhancedPhysicsCNN()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 2. 加载数据
    # (假设已通过data_generator生成数据)
    train_loader, val_loader, _ = create_dataloaders(...)
    
    # 3. 训练循环
    best_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(100):
        model.train()
        epoch_train_loss = 0
        
        # 训练阶段
        for features, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                epoch_val_loss += criterion(outputs.squeeze(), targets).item()
        
        # 记录损失
        train_loss = epoch_train_loss/len(train_loader)
        val_loss = epoch_val_loss/len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_loss:
            torch.save(model.state_dict(), "best_model.pth")
            best_loss = val_loss
        
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
    
    # 可视化训练过程
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('training_curve.png')

if __name__ == "__main__":
    train()