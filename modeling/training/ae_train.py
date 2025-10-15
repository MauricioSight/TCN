import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from modeling.training.early_stopping import EarlyStopping
from modeling.training.pytorch_base import PytorchTrainingAlgorithm
from modeling.structure.pytorch_base import PytorchModelStructure
from utils.create_loader import create_loader
from utils.criterion import get_criterion
from utils.seed_all import DEFAULT_SEED


class AETrain(PytorchTrainingAlgorithm):
    def fit(self, model: PytorchModelStructure, train_loader: DataLoader, epoch: int) -> float:
        """"
        Update model weights

        args:
            model: pytorch model
            train_loader: data
            epoch: current epoch

        returns:
            train loss
        """

        model.train()
        train_loss = 0

        for i, (data, target, _) in enumerate(train_loader):
            self.optimizer.zero_grad()
            out = model.forward(data)
            loss = self.criterion(out, target)
            loss.backward()

            train_loss += loss.item()

            self.optimizer.step()

            # metrics logs
            if i % 100 == 0 or i == len(train_loader) - 1:
                self.logger.info('Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,  i * train_loader.batch_size, len(train_loader.sampler), 
                    100. * i / len(train_loader), loss.item()))
        
        train_loss = train_loss / len(train_loader)

        return train_loss
    

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

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (data, target, _) in enumerate(val_loader):
                out = model.forward(data)
                loss = self.criterion(out, target)
                val_loss += loss.item()

                if i % 100 == 0 or i == len(val_loader) - 1:
                    self.logger.info('Epoch: {} \t[{}/{} ({:.0f}%)]\tValidation loss: {:.6f}'.format(
                        epoch, i * val_loader.batch_size, len(val_loader.sampler), 100. * i / len(val_loader), loss.item()))

        val_loss = val_loss / len(val_loader)

        return val_loss


    def __create_loaders(self, X: np.ndarray, y: pd.DataFrame) -> tuple[DataLoader, DataLoader]:
        y = y.reset_index()

        # Pred is an unsupervised method. So, just benign samples in training phase
        benign_idx = y[y['label'] == 'Normal'].index.to_list()

        train_idx, val_idx = train_test_split(benign_idx, train_size=0.8, random_state=DEFAULT_SEED, shuffle=True)

        self.logger.info(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}")

        self.logger.info(f"Train labels: \n{y.iloc[train_idx]['label'].value_counts()}")
        self.logger.info(f"Validation labels: \n{y.iloc[val_idx]['label'].value_counts()}")

        # In prediction, the target is the input shifted by 1
        data = [[X[i], X[i], i] for i in range(X.shape[0])]

        g = torch.Generator()
        g.manual_seed(42)

        batch_size = self.config.get('modeling', {}).get('training', {}).get('batch_size')

        train_loader = create_loader(data, train_idx, batch_size, self.device, g)
        val_loader = create_loader(data, val_idx, batch_size, self.device, g)
        
        return train_loader, val_loader


    def train(self, model: PytorchModelStructure, X: np.ndarray, y: pd.DataFrame) -> tuple[float, float]:
        """"
        Execute training. Early stopping is options based on configs

        args:
            model: pytorch model
            train_loader: train data
            val_loader: validation data

        returns:
            train loss, validation loss
        """

        criterion_name  =   self.config.get('modeling', {}).get('training', {}).get('criterion')
        reduction       =   self.config.get('modeling', {}).get('training', {}).get('reduction', 'mean')
        learning_rate   =   self.config.get('modeling', {}).get('training', {}).get('learning_rate')
        num_epochs      =   self.config.get('modeling', {}).get('training', {}).get('num_epochs')
        es_patience     =   self.config.get('modeling', {}).get('training', {}).get('es_patience')

        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        self.criterion = get_criterion(criterion_name, reduction=reduction)

        early_stopping = None
        if es_patience:
            early_stopping = EarlyStopping(self.config, self.logger, model)


        train_loader, val_loader = self.__create_loaders(X, y)

        self.logger.info(f"Running for {num_epochs} epochs")
        self.logger.info(f"-------------------- Training started -------------------")
        for epoch in range(num_epochs):
            train_loss = self.fit(model, train_loader, epoch)
            val_loss =  self.validate(model, val_loader, epoch)

            self.tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if early_stopping:
                condition_match = early_stopping(val_loss, epoch)
                if condition_match:
                    break
            else:
                model.save_model_state_dict()
            
        return train_loss, val_loss
