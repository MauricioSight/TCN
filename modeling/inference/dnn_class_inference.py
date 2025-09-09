import torch
import numpy as np
import pandas as pd

from modeling.inference.pytorch_base import PytorchInference
from modeling.structure.pytorch_base import PytorchModelStructure
from utils.create_loader import create_loader
from utils.criterion import get_criterion

class DNNClassInference(PytorchInference):
    def inference(self, model: PytorchModelStructure, X: np.ndarray, y: pd.DataFrame) -> tuple[np.ndarray, float]:
        """"
        Inference model to get validation loss

        args:
            model: pytorch model
            X: data
            y: labels

        returns:
            predictions and loss
        """
        criterion_name  =   self.config.get('modeling', {}).get('training', {}).get('criterion')
        reduction       =   self.config.get('modeling', {}).get('training', {}).get('reduction', 'mean')
        batch_size      =   self.config.get('modeling', {}).get('inference', {}).get('batch_size')

        self.criterion = get_criterion(criterion_name, reduction=reduction)

        model.eval()
        test_loss = 0
        y_scores = []
        y_true = []

        g = torch.Generator()
        g.manual_seed(42)
        
        data = [[X[i], y.iloc[i]['label'], i] for i in range(X.shape[0])]
        self.logger.info(f"Testing labels: \n{y['label'].value_counts()}")
        data_loader = create_loader(data, None, batch_size, self.device, g)

        with torch.no_grad():
            for i, (data, target, label_idx) in enumerate(data_loader):
                out = model.forward(data)
                loss = self.criterion(out, target)
                test_loss += loss.item()

                y_scores.extend(out.cpu().numpy())
                y_true.extend(y.iloc[label_idx]['label'])

                if i % 10000 == 0 or i * len(data) == len(data_loader.dataset) - 1:
                    self.logger.info('Test loss: {:.6f} \t[{}/{} ({:.0f}%)]'.format(
                        loss.item(), i * len(data), len(data_loader.dataset), 100. * i / len(data_loader)))

        test_loss = test_loss / len(data_loader)

        return y_true, y_scores, test_loss
