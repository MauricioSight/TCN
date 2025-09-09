import torch
import torch.nn as nn
from torch import device as TorchDevice
import numpy as np
import pandas as pd

from logging import Logger
from utils.experiment_io import get_run_dir

class PytorchModelStructure(nn.Module):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        super(PytorchModelStructure, self).__init__()

        self.config = config
        self.logger = logger
        self.device = device
        self.run_id = config.get('run_id')
        self.run_dir = get_run_dir(self.run_id)
        self.model_dir = self.run_dir / "model.pt"

    def forward(self, x):
        raise NotImplementedError("Each model must implement the forward pass.")
    

    def save_model_state_dict(self):
        torch.save(self.state_dict(), self.model_dir)


    def load_model_state_dict(self):
        if self.model_dir.exists():
            self.load_state_dict(torch.load(self.model_dir, map_location=self.device))

    def compile(self):
        self = self.to(device=self.device)
        self.load_model_state_dict()


    def featuring(self, values: np.ndarray, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return values, labels
