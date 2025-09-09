import torch
import numpy as np
import pandas as pd
from typing import Tuple

import torchvision
import torchvision.transforms as transforms

from data_loader.base import DataLoader


class MNISTLoader(DataLoader):
    def load(self) -> Tuple[np.ndarray, pd.DataFrame]:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        dataset = torchvision.datasets.MNIST(root='./data', train=self.config.get('phase') == 'train', transform=transform, download=True)

        labels = dataset.targets  # torch.Tensor
        labels_df = pd.DataFrame(labels.numpy(), columns=["label"])

        return dataset.data.to(dtype=torch.float32), labels_df