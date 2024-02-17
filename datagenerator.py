import numpy as np
from collections import defaultdict
import datetime
import pandas as pd


class DataGenerator():
    """
    Generates data for transaction fraud
    """
    def __init__(self,
                n_features:int=4,
                reset_prob:float=0.001) -> None:
        
        self.n_features = n_features
        self.reset_prob = reset_prob
        self.feature_dist_params = {}
        self.coeffs = np.empty((3,3))
        self.reset_feature_params()
        
    def reset_feature_params(self):
        feature_dist_params = {}
        for i in range(self.n_features):
            a = np.random.uniform(1.0, 5.0)
            if np.random.rand() > 0.5:
                b = a/np.random.uniform(0.5, 2.0)
            else:
                b = a * np.random.uniform(2.0, 5.0)

            feature_dist_params[f'a{i}'] = a
            feature_dist_params[f'b{i}'] = b

        self.feature_dist_params = feature_dist_params
        self.coeffs = np.random.uniform(0.5, 2.0, 4)

    def generate_data(self, n_records:int) -> list:
        """
        Generate fraud probabilities across features.

        Parameters:
        length (float): The length of the rectangle.
        width (float): The width of the rectangle.

        Returns:
        float: The area of the rectangle.

        Example:
        >>> calculate_area(5.0, 3.0)
        15.0
        """
        if np.random.rand() < self.reset_prob:
            self.reset_feature_params()

        x = np.zeros((n_records, self.n_features))
        x[:, 0] =  np.random.beta(self.feature_dist_params['a0'], self.feature_dist_params['b0'], n_records)
        x[:, 1] =  np.random.beta(self.feature_dist_params['a1'], self.feature_dist_params['b1'], n_records)
        x[:, 2] =  np.random.beta(self.feature_dist_params['a2'], self.feature_dist_params['b2'], n_records)
        x[:, 3] =  np.random.beta(self.feature_dist_params['a3'], self.feature_dist_params['b3'], n_records)

        epsilon = np.random.rand(n_records) * 50.0
        y = np.dot(x, self.coeffs) * 100.0 + epsilon
        timestamp = datetime.datetime.now()

        df = pd.DataFrame(x)
        df.columns = [f'x{i}' for i in range(self.n_features)]
        df['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M')
        df['y'] = y

        return df


if __name__ == '__main__':
    dg = DataGenerator(n_features=4)
    df = dg.generate_data(1000)
    print(df.head(10))
    print(df['y'].mean())
        