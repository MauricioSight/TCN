from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from logging import Logger

class InferenceMetrics(ABC):
    def __init__(self, config: dict, logger: Logger):
        self.config = config
        self.logger = logger

    @abstractmethod
    def get_overall_metrics(self, y_true: pd.DataFrame, y_scores: np.ndarray) -> dict:
        """"
        Get overall metrics

        args:
            y_true: Data frame with labels
            y_scores: Model's output
            threshold: The threshold

        returns:
            dict of metrics
        """
        pass


    @abstractmethod
    def get_threshold(self, *args) -> float:
        """
        Define ou load threshold

        returns:
            threshold
        """
        pass