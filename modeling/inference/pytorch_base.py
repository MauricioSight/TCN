from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from torch import device as TorchDevice
from modeling.structure.pytorch_base import PytorchModelStructure
from logging import Logger

class PytorchInference(ABC):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        self.config = config
        self.logger = logger
        self.device = device

    @abstractmethod
    def inference(self, model: PytorchModelStructure, X: np.ndarray, y: pd.DataFrame) -> tuple[np.ndarray, float]:
        """
        Runs pytorch model

        args:
            model: the pytorch model
            X: data
            y: labels

        returns:
            the model output and loss
        """
        pass
