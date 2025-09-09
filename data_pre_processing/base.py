from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from logging import Logger
from data_loader.base import DataLoader

class DataPrePrecessing(ABC):
    def __init__(self, config: dict, logger: Logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def initialize(self, data_loader: DataLoader) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        verify if has saved processed data, if not run process and save

        args:
            data_loader: raw data loader

        returns:
            Tuple: (X: data, y: labels)
        """
        pass

    @abstractmethod
    def process(self, data: np.ndarray, target: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        process raw data

        args:
            data: raw,
            target: raw

        returns:
            Tuple: (X: data processed, y: target processed)
        """
        pass

    @abstractmethod
    def save(self, data: np.ndarray, target: pd.DataFrame):
        """
        save processed data

        args:
            X: data processed,
            y: target processed,
        """
        pass

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        load processed data

        returns:
            Tuple: (X: data processed, y: target processed)
        """
        pass

