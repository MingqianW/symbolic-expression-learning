# model.py
import torch
import torch.nn as nn

class PhysicsCNN(nn.Module):
    """
    CNN架构专为学习 X_{t+1} = X_t - X_t/(R*C) 设计
    输入维度: [batch_size, 3] (R, C, X_t)
    输出维度: [batch_size, 1] (X_t+1)
    """
    def __init__(self):
        super(PhysicsCNN, self).__init__()
        
        # 输入形状: [batch, 1, 3] (添加通道维度)
        self.conv_layers = nn.Sequential(
            # 1D卷积学习局部特征关系
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            # 深度可分离卷积增强特征提取
            nn.Conv1d(16, 32, kernel_size=3, padding=1, groups=16),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.AdaptiveAvgPool1d(1)  # 全局池化
        )
        
        # 全连接层进行物理关系建模
        self.fc_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 物理引导的初始化
        self._initialize_weights()

    def forward(self, x):
        # 原始输入维度: [batch, 3]
        x = x.unsqueeze(1)  # 添加通道维度 -> [batch, 1, 3]
        
        # 特征提取
        conv_out = self.conv_layers(x)  # [batch, 32, 1]
        conv_out = conv_out.view(conv_out.size(0), -1)  # [batch, 32]
        
        # 物理关系回归
        return self.fc_layers(conv_out)
    
    def _initialize_weights(self):
        """物理启发式初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 最后层初始化为近似恒等变换
                if m.out_features == 1:
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight)

class EnhancedPhysicsCNN(PhysicsCNN):
    """
    增强版架构，添加残差连接和物理约束
    """
    def __init__(self):
        super().__init__()
        
        # 残差分支
        self.res_fc = nn.Sequential(
            nn.Linear(3, 16),  # 直接处理原始输入
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        # 原始特征路径
        conv_out = super().forward(x)
        
        # 残差路径
        identity = self.res_fc(x)
        
        # 组合输出
        return conv_out + identity * 0.3  # 控制残差强度