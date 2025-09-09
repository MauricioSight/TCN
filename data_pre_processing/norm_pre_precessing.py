import os
import pickle
import random
from typing import Tuple

from data_loader.base import DataLoader
import torch
import numpy as np
import pandas as pd

from data_pre_processing.base import DataPrePrecessing


class NormPrePrecessing(DataPrePrecessing):
    def initialize(self, data_loader: DataLoader) -> Tuple[np.ndarray, pd.DataFrame]:
        path = self.get_output_path()

        if not os.path.exists(path):
            data, target = data_loader.load()
            X, y = self.process(data, target)
            self.save(X, y)
            return X, y
        
        return self.load(path)


    def process(self, data: np.ndarray, target: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        return data, target


    def save(self, X: np.ndarray, y: pd.DataFrame):
        self.logger.info("Saving data in cash file...")

        path = self.get_output_path()

        torch.save({'X': X, 'y': y}, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Data saved to {path}")
    

    def load(self, path: str) -> tuple[np.ndarray, pd.DataFrame]:
        self.logger.info(f"Loading cached data from: {path}")
        cache = torch.load(path, weights_only=False)
        X, y = cache['X'], cache['y']


        load_subset = self.config.get('pre_processing', {}).get('load_subset')
        if load_subset is not None:
            self.logger.warning(f"Loading data with subset of {load_subset}%")

            indices = np.random.choice(len(X), size=int(load_subset*len(X)), replace=False)
            X = X[indices]
            y = y.iloc[indices].reset_index(drop=True)


        return X, y
    
    
    def get_output_path(self) -> str:
        """
        Get the output path for the processed data.
        """
        phase = self.config.get('phase')
        processed_path = self.config.get('pre_processing', {}).get('processed_path')
        method = self.config.get('pre_processing', {}).get('name')

        os.makedirs(processed_path, exist_ok=True)
        return f"{processed_path}/{phase}_{method}.pt"
    
