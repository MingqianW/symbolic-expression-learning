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
                 initial_x_range: tuple = (0.0, 5.0),
                 noise_scale: float = 1.0):
        """
        
        :param f: dynamic function f(x, R, C)
        :param R_range: the sample range of R (min, max)
        :param C_range: the sample range of C (min, max)
        :param noise_scale: std of Gaussian noise term
        :param initial_x_range: the sample range of initial x(min, max)
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
        self.initial_x_range = initial_x_range
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
            
        for _ in range(num_sequences): #Fixed::note for now, each sequence has same R and C. In other words, R and C are essentially "constant" in each sequence.
            X0 = self._random_state.uniform(*self.initial_x_range)
            X = [X0]
            R_list = []
            C_list = []
            I_list = []
            
            for _ in range(num_points - 1):
                R_t = self._random_state.uniform(*self.R_range)
                C_t = self._random_state.uniform(*self.C_range)
                I_t = self._random_state.normal(scale=self.noise_scale)
                R_list.append(R_t)
                C_list.append(C_t)     
                I_list.append(I_t)            
                current_x = X[-1]
                delta = self.f(current_x, R_t, C_t, I_t)
                X.append(delta)
                
            
            self.sequences.append({
                'R': np.array(R_list),
                'C': np.array(C_list),
                'X': np.array(X),
                'I': np.array(I_list),
            })
            
    def get_all_parameters(self, 
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        random_seed: int = None):
        return [(seq['R'], seq['C'], seq['I']) for seq in self.sequences]

    def clear_data(self):
        self.sequences = []
        
    def prepare_datasets_by_col(self, 
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.1,
                        random_seed: int = None):
        """
        return the data as X:[N,4] and Y as [N,1]
        """
        features, targets = [], []
        for seq in self.sequences:
            X = seq['X']
            R = seq['R']
            C = seq['C']
            I = seq['I']
            # (R, C, X_t) -> X_{t+1}
            for t in range(len(X)-1):
                features.append([R[t], C[t], X[t], I[t]])
                targets.append(X[t+1])
                
        X = torch.tensor(features)
        y = torch.tensor(targets)
        X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_seed)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size= (1 - train_ratio - val_ratio) /(1 - train_ratio), random_state=random_seed)
        return X_train, y_train,X_test,y_test
    
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
            I = seq['I']
            # (R, C, X_t) -> X_{t+1}
            for t in range(len(X)-1):
                features.append([R[t], C[t], X[t], I[t]])
                targets.append(X[t+1])
                
        X = np.array(features)
        y = np.array(targets)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1 - train_ratio), random_state=random_seed)
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size= (1 - train_ratio - val_ratio) /(1 - train_ratio), random_state=random_seed)
        
        train_dataset = SequencesDataset(X_train, y_train)
        val_dataset = SequencesDataset(X_val, y_val)
        test_dataset = SequencesDataset(X_test, y_test)
        
        return train_dataset, val_dataset, test_dataset
    
if __name__ == "__main__":
    def f(x, R, C,I):
        return x /(R*C) +I  # Example f
    
    generator = DataGenerator(
        f = f,
        R_range=(1.0, 5.0),
        C_range=(1.0, 5.0),
        initial_x_range=(0.0, 10.0),
        noise_scale=0.01
    )
    
    # Generate sample data
    generator.generate(
        num_points=5,
        num_sequences=2,
        random_seed=42
    )
    
    # Verify data structure
    print("Generated sequences:")
    for i, seq in enumerate(generator.sequences):
        print(f"Sequence {i+1}:")
        print(f"Initial X0: {seq['X'][0]:.2f}")
        print(f"R values: {seq['R'].round(2)}")
        print(f"C values: {seq['C'].round(2)}")
        print(f"I values: {seq['I'].round(2)}")
        print(f"X progression: {seq['X'].round(2)}")
        print()
    
    # Verify calculations
    print("Step-by-step validation:")
    for seq in generator.sequences:
        X = seq['X']
        R = seq['R']
        C = seq['C']
        I = seq['I']
        for t in range(len(X)-1):
            x_t = X[t]
            r_t = R[t]
            c_t = C[t]
            I_t = I[t]
            expected = f(r_t, c_t,x_t,I_t)
            actual = X[t+1]
            noise = actual - expected
            
            print(f"t={t}:")
            print(f"  expected: {expected:.4f}")
            print(f"  actual: {actual:.4f}")
            print("-"*60)
        print("\n"*2)
        
    X_train, y_train, X_test, y_test = generator.prepare_datasets_by_col(train_ratio=0.8, val_ratio=0.1, random_seed=42)
    print("Training data:")
    print(X_train)
    print(y_train)
    print("Test data:")
    print(X_test)
    print(y_test)
    
    # Validate that the data matches the function
    print("Validation of generated data:")
    for i in range(len(X_train)):
        R, C, X, I = X_train[i]
        expected = f( R.item(), C.item(),X.item(), I.item())
        actual = y_train[i].item()
        print(f"Train data point {i}:")
        print(f"  Input: R={R.item()}, C={C.item()}, X={X.item()}, I={I.item()}")
        print(f"  Expected: {expected:.4f}")
        print(f"  Actual: {actual:.4f}")
        print("-"*60)
    
    for i in range(len(X_test)):
        R, C, X, I = X_test[i]
        expected = f( R.item(), C.item(),X.item(), I.item())
        actual = y_test[i].item()
        print(f"Test data point {i}:")
        print(f"  Input: R={R.item()}, C={C.item()}, X={X.item()}, I={I.item()}")
        print(f"  Expected: {expected:.4f}")
        print(f"  Actual: {actual:.4f}")
        print("-"*60)