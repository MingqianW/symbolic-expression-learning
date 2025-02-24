import numpy as np
import pandas as pd

class DataGenerator:
    """
    Data generator that follows X_{t+1} = X_t + f(X_t, R, C) + I, where i is some Gaussian noise term
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
        """
        self.f = f
        self.R_range = R_range
        self.C_range = C_range
        self.noise_scale = noise_scale
        self.initial_x = initial_x
        
        # storaged of generated data
        self.X = None
        self.R = None
        self.C = None
        
        # random_state for reproducibility
        self._random_state = None

    def generate(self, 
                 num_points: int,
                 reset: bool = True,
                 random_seed: int = None) -> None:
        """
        generate new data
        
        :param num_points: the amount of new data point that we want to generate
        :param reset: whether to reset current data sequence 
        :param random_seed: random_seed
        """
        self._init_random_state(random_seed)
        
        self.R = np.random.uniform(*self.R_range)
        self.C = np.random.uniform(*self.C_range)
        
        if reset:
            X = [self.initial_x]
        elif self.X:
            X = [self.X[-1]] #continue and initialize with the last element of previous data sequence
        else:
            X = [self.initial_x]
        
        # generation
        for _ in range(num_points - 1):
            current_x = X[-1]
            delta = self.f(current_x, self.R, self.C)
            noise = np.random.normal(scale=self.noise_scale)
            X.append(current_x + delta + noise)
            
        if reset:
            self.X = np.array(X)  
        else:
            np.concatenate([self.X, np.array(X[1:])])

    def save_csv(self, filename: str) -> None:
        if self.X is None:
            raise ValueError("No data available. Call generate() first.")
        pd.DataFrame({"X": self.X}).to_csv(filename, index=False)

    def get_parameters(self) -> dict:
        """return the current parameters we are using"""
        return {
            "R": self.R,
            "C": self.C,
            "R_range": self.R_range,
            "C_range": self.C_range,
            "noise_scale": self.noise_scale,
            "initial_x": self.initial_x
        }

    def _init_random_state(self, seed: int = None) -> None:
        if seed is not None:
            self._random_state = np.random.RandomState(seed)
        else:
            self._random_state = np.random.RandomState()

    def __len__(self) -> int:
        return len(self.X) if self.X is not None else 0

###################Sample use############################
if __name__ == "__main__":
    def rc_dynamics(x, R, C):
        return -x / (R * C)

    generator = DataGenerator(
        f=rc_dynamics,
        R_range=(1.0, 10.0),
        C_range=(0.1, 1.0),
        noise_scale=0.05,
        initial_x=5.0
    )

    generator.generate(100, random_seed=42)
    
    
    
   