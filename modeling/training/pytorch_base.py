from abc import ABC, abstractmethod

from torch import device as TorchDevice
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from logging import Logger
from tracker.base_tracker import BaseTracker
from modeling.structure.pytorch_base import PytorchModelStructure


class PytorchTrainingAlgorithm(ABC):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice, tracker: BaseTracker):
        self.config = config
        self.logger = logger
        self.device = device
        self.tracker = tracker

    @abstractmethod
    def fit(self, model: PytorchModelStructure, train_loader: DataLoader, epoch: int) -> float:
        """
        Update model weights

        args:
            model: pytorch model
            train_loader: data
            epoch: current epoch

        returns:
            train loss
        """
        pass


    @abstractmethod
    def validate(self, model: PytorchModelStructure, val_loader: DataLoader, epoch: int) -> float:
        """"
        Inference model to get validation loss

        args:
            model: pytorch model
            val_loader: data
            epoch: current epoch

        returns:
            validation loss
        """
        pass


    @abstractmethod
    def train(self, model: PytorchModelStructure, X: np.ndarray, y: pd.DataFrame) -> tuple[float, float]:
        """"
        Execute training. Early stopping is optional based on configs

        args:
            model: pytorch model
            X: data
            y: labels

        returns:
            train loss, validation loss
        """
        pass
