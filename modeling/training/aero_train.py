import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

from modeling.training.early_stopping import EarlyStopping
from modeling.training.pytorch_base import PytorchTrainingAlgorithm
from modeling.structure.aero import AERO
from utils.create_loader import create_loader
from utils.criterion import get_criterion
from utils.seed_all import DEFAULT_SEED


class AEROTrain(PytorchTrainingAlgorithm):
    def fit(self, model: AERO, train_loader: DataLoader, epoch: int) -> float:
        """"
        Update model weights

        args:
            model: pytorch model
            train_loader: data
            epoch: current epoch

        returns:
            train loss
        """
        in_channels = self.config.get('modeling', {}).get('structure', {}).get('in_channels')

        model.train()
        train_loss = 0

        for i, (data, _, _) in enumerate(train_loader):
            T = data[:, :9]
            P = data[:, 9:-9]
            S = data[:, -9:]

            P = P.reshape(data.shape[0], in_channels, -1)
            
            # Train the autoencoder
            self.ae_optimizer.zero_grad()

            emb             = model.encoder(T, P, S)
            S_t, P_t, T_t   = model.decoder(emb)
            loss            = self.ae_criterion(S, P, T, S_t, P_t, T_t)
            
            loss.backward()
            self.ae_optimizer.step()
            
            train_loss += loss.item()

            # metrics logs
            if i % 100 == 0 or i == len(train_loader) - 1:
                self.logger.info('Epoch: {} \t[{}/{} ({:.0f}%)]\tAE Loss: {:.6f}'.format(
                    epoch,  i * train_loader.batch_size, len(train_loader.sampler), 
                    100. * i / len(train_loader), loss.item()))
        
        train_loss = train_loss / len(train_loader)

        return train_loss
    
    
    def train_point_mapper(self, model: AERO, train_loader: DataLoader, epoch: int):
        model.train()
        train_pm_loss = 0
        in_channels = self.config.get('modeling', {}).get('structure', {}).get('in_channels')

        for batch_idx, (data, _, _) in enumerate(train_loader):
            T = data[:, :9]
            P = data[:, 9:-9]
            S = data[:, -9:]

            P = P.reshape(data.shape[0], in_channels, -1)

            self.pm_optimizer.zero_grad()

            # Step 1: Encoding
            emb = model.encoder(T, P, S)  # H ← φ_E(X, θ_E)

            # Step 2: Mapping
            mapped_points = model.point_mapper(emb)  # M ← φ_M(H, θ_M)

            # Step 3: Column-wise mean (criterion point)
            criterion_point = mapped_points.mean(dim=0, keepdim=True)  # m̄ ← mean(M)

            # Step 4: Compute pretraining loss
            pm_loss = torch.sum(torch.sum((mapped_points - criterion_point) ** 2), dim=0)

            # Step 5: Backpropagate and optimize
            pm_loss.backward()
            self.pm_optimizer.step()

            train_pm_loss += pm_loss.item()

            if batch_idx % 1000 == 0 or batch_idx * len(data) == len(train_loader.dataset) - 1:
                self.logger.info('Epoch: {} \t[{}/{} ({:.0f}%)]\tPM Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), pm_loss.item()))

        train_pm_loss /= len(train_loader)
        return train_pm_loss

    def fine_tune_point_mapper(self, model: AERO, train_loader: DataLoader, epoch: int):
        """
        Fine-tune point mapper after pretraining.
        
        Args:
            train_loader: DataLoader for training data.
            num_epochs: Number of epochs for fine-tuning.
        
        Returns:
            avg_fine_tune_loss: Average fine-tuning loss.
        """
        in_channels = self.config.get('modeling', {}).get('structure', {}).get('in_channels')

        model.train()
        fine_tune_loss = 0

        for batch_idx, (data, _, _) in enumerate(train_loader):
            T = data[:, :9]
            P = data[:, 9:-9]
            S = data[:, -9:]

            P = P.reshape(data.shape[0], in_channels, -1)

            self.pm_optimizer.zero_grad()

            # Step 1: Encode
            emb = model.encoder(T, P, S)  # H ← φ_E(X, θ_E)

            # Step 2: Map
            mapped_points = model.point_mapper(emb)  # M ← φ_M(H, θ_M)

            # Step 3: Compute fine-tuning loss
            pm_loss = torch.sum(torch.sum((mapped_points - model.criterion_point) ** 2, dim=1), dim=0)  # L_M using Equation (5)

            # Step 4: Backprop and optimize
            pm_loss.backward()
            self.pm_optimizer.step()

            fine_tune_loss += pm_loss.item()

            if batch_idx % 1000 == 0 or batch_idx * len(data) == len(train_loader.dataset) - 1:
                self.logger.info('Epoch: {} \t[{}/{} ({:.0f}%)]\tFine-tune Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), pm_loss.item()))

        fine_tune_loss /= len(train_loader)
        return fine_tune_loss

    def validate(self, model: AERO, val_loader: DataLoader, epoch: int) -> float:
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
    

    def ae_criterion(self, S, P, T, S_t, P_t, T_t):
        # reconstruction loss is measured as the sum of the mean squared errors of the three features
        loss_s = F.mse_loss(S_t, S)
        loss_p = F.mse_loss(P_t, P)
        loss_t = F.mse_loss(T_t, T)
        return loss_s + loss_p + loss_t


    def train(self, model: AERO, X: np.ndarray, y: pd.DataFrame) -> tuple[float, float]:
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
        ae_num_epochs   =   self.config.get('modeling', {}).get('training', {}).get('ae_num_epochs')
        pm_num_epochs   =   self.config.get('modeling', {}).get('training', {}).get('pm_num_epochs')
        ft_num_epochs   =   self.config.get('modeling', {}).get('training', {}).get('ft_num_epochs')

        self.ae_optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-4)
        self.pm_optimizer = torch.optim.Adam(model.point_mapper.parameters(), lr=1e-5)
        self.criterion = get_criterion(criterion_name, reduction=reduction)

        train_loader, _ = self.__create_loaders(X, y)

        self.logger.info(f"Running for {ae_num_epochs} epochs")
        self.logger.info(f"-------------------- Training AE started -------------------")
        for epoch in range(ae_num_epochs):
            train_loss = self.fit(model, train_loader, epoch)

            self.tracker.log_metrics({"ae_train_loss": train_loss}, step=epoch)

            model.save_model_state_dict()


        self.logger.info(f"-------------------- Training Point Mapper started -------------------")
        for epoch in range(pm_num_epochs):
            train_loss = self.train_point_mapper(model, train_loader, epoch)

            self.tracker.log_metrics({"pm_train_loss": train_loss}, step=ae_num_epochs+epoch)

            model.save_model_state_dict()


        model.compute_criterion_point(train_loader)


        self.logger.info(f"-------------------- Fine tunning Point Mapper started -------------------")
        for epoch in range(ft_num_epochs):
            train_loss = self.fine_tune_point_mapper(model, train_loader, epoch)

            self.tracker.log_metrics({"pm_ft_loss": train_loss}, step=ae_num_epochs+pm_num_epochs+epoch)

            model.save_model_state_dict()
            
        return train_loss, -1
