# main.py
import numpy as np
from data_generator import DataGenerator
from SimpleCNN.train import train_simple_model

def dynamic_func(x, R, C):
    return x / (1+ R*C )

generator = DataGenerator(
    f=dynamic_func,
    R_range=(1.0, 5.0),
    C_range=(1.0, 5.0),
    initial_x_range=(-2.0, 2.0),
    noise_scale=0.1
)
generator.generate(num_points=100, num_sequences=200, random_seed=42)

train_dataset, val_dataset, test_dataset = generator.prepare_datasets(
    train_ratio=0.8,
    val_ratio=0.1,
    random_seed=42
)

if __name__ == "__main__":
    train_simple_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )