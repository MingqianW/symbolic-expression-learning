import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from data_loader import SequencesDataset
class DataGenerator:
    """
    Data generator that follows X_{t+1} = X_t + f(X_t, R, C) + I, where i is some Gaussian noise term
    Generate several indepent sequence of data, which have same sturcture: X_{t+1} = X_t + f(X_t, R, C) + I (Note: we enforce f(X_t, R, C) be same)
    but different X_0 and diffrent randomly generated R,C,I
    """
    
    def __init__(self, 
                 f: callable,
                 R_range: tuple = (1.0, 10.0),
                 C_range: tuple = (1.0, 10.0),
                 noise_scale: float = 1.0,
                 initial_x: float = 0.0):
        """
        
        :param f: dynamic function f(x, R, C)
        :param R_range: the sample range of R (min, max)
        :param C_range: the samole range of C (min, max)
        :param noise_scale: std of Gaussian noise term
        :param initial_x: initial value of X_0
        :param sequences: sequence of {
                'R': R,
                'C': C,
                'X': X
            }
        """
        self.f = f
        self.R_range = R_range
        self.C_range = C_range
        self.noise_scale = noise_scale
        self.initial_x = initial_x
        self.sequences = []
        
        # storaged of generated data
        self.sequences = []
        self._current_sequence = None
        self._random_state = np.random.RandomState()
        
        # random_state for reproducibility
        self._random_state = None
    def generate(self, 
                 num_points: int,
                 num_sequences: int = 1,
                 random_seed: int = None):
        """
        generate multiple new sequence of data(default is to generate one sequence)
        
        :param num_points: the amount of new data point that we want to generate
        :param reset: whether to reset current data sequence 
        :param random_seed: random_seed
        """
        if random_seed is not None:
            self._random_state = np.random.RandomState(random_seed)
            
        for _ in range(num_sequences):
            R = self._random_state.uniform(*self.R_range)
            C = self._random_state.uniform(*self.C_range)
            X = [self.initial_x]
            
            for _ in range(num_points - 1):
                current_x = X[-1]
                delta = self.f(current_x, R, C)
                noise = self._random_state.normal(scale=self.noise_scale)
                X.append(current_x + delta + noise)
            
            self.sequences.append({
                'R': R,
                'C': C,
                'X': np.array(X)
            })
    def get_all_parameters(self):
        return [(seq['R'], seq['C']) for seq in self.sequences]

    def clear_data(self):
        self.sequences = []
        
    def prepare_datasets(self, 
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        random_seed: int = None):
        """
        prepare data to train/val/test as 8:1:1
        return them as pytorch dataset
        """
        features, targets = [], []
        for seq in self.sequences:
            X = seq['X']
            R = seq['R']
            C = seq['C']
            
            # (R, C, X_t) -> X_{t+1}
            for t in range(len(X)-1):
                features.append([R, C, X[t]])
                targets.append(X[t+1])
                
        X = np.array(features)
        y = np.array(targets)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_ratio), random_state=random_seed)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_seed)
        
        train_dataset = SequencesDataset(X_train, y_train)
        val_dataset = SequencesDataset(X_val, y_val)
        test_dataset = SequencesDataset(X_test, y_test)
        
        return train_dataset, val_dataset, test_dataset

