import torch
from torch.utils.data import DataLoader

import torch
import numpy as np
from torch.utils.data import DataLoader
import os

class SequencesDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        """
        :param features: (R, C, X_t)
        :param targets: X_{t+1}
        :param scaler: optional scaler
        """
        features_np = np.array(features) if not isinstance(features, np.ndarray) else features
        targets_np = np.array(targets) if not isinstance(targets, np.ndarray) else targets
        
        # Normalization
        self.features = torch.FloatTensor(features_np)          
        self.targets = torch.FloatTensor(targets_np)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    def save(self, filename):
        np.savez(filename, 
                 features=self.features.numpy(),
                 targets=self.targets.numpy())
        
    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        return cls(
            features=data['features'],
            targets=data['targets']
        )

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    shuffle_train: bool = True,
    save_dir: str = None
) -> tuple:

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size*2,
        shuffle=False,
        pin_memory=True
    )
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        train_dataset.save(os.path.join(save_dir, 'train.npz'))
        val_dataset.save(os.path.join(save_dir, 'val.npz'))
        test_dataset.save(os.path.join(save_dir, 'test.npz'))
    
    return train_loader, val_loader, test_loader
