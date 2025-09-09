import numpy as np
import pandas as pd
from typing import Tuple

from logging import Logger
from abc import ABC, abstractmethod

class DataLoader(ABC):
    def __init__(self, config: dict, logger: Logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        load raw data

        returns:
            Tuple: (X: data, y: labels)
        """
        pass

