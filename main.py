# main.py
from data_generator import DataGenerator
from data_loader import create_dataloaders, SequencesDataset
import torch

def main():
    def rc_inverse(x, R, C):
        return x / (1 + (R * C))

    generator = DataGenerator(
        f=rc_inverse,
        R_range=(1.0, 10.0),
        C_range=(0.1, 1.0),
        noise_scale=0.05,
        initial_x=5.0   
    )

    # Generate data
    # Generate 100 sequences while each sequence has 50 points
    # Total amount of sample: 100 * (50 - 1) = 4900 since we do not have the 
    generator.generate(
        num_points=50,        
        num_sequences=100,     
        random_seed=42        # for reproducibility
    )

    # 4. Prepare data set for training
    # X: [R, C, X_t] -> (amount of sample, 3) = (4900,3)
    # y: X_t+1       -> (amount of sample,) = (4900,)
    train_set, val_set, test_set = generator.prepare_datasets(
        train_ratio=0.8,
        val_ratio=0.1,
        random_seed=42
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_set,
        val_dataset=val_set,
        test_dataset=test_set,
        batch_size=64,        
        shuffle_train=True,   
        save_dir='./data'      
    )


if __name__ == "__main__":
    main()