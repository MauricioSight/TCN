from logging import Logger
from modeling.structure.pytorch_base import PytorchModelStructure

class EarlyStopping:
    def __init__(self, config, logger: Logger, model: PytorchModelStructure):
        self.config = config
        self.logger = logger
        self.model = model
        
        self.training_config = config.get('training')
        self.learning_rate = self.training_config.get('learning_rate')
        self.early_stopping_patience = self.training_config.get('early_stopping_patience')
        self.early_stopping_delta = self.training_config.get('early_stopping_delta', 0.0001)

        self.best_val_loss = float("inf")


    def __check_early_stopping(self, val_loss: float) -> int:
        """
        Check if early stopping condition match

        args:
            val_loss: validation loss

        returns:
            condition_match: True if not matched and -1 if does
        """
        condition_match = False
        # Early stopping update
        if val_loss < self.best_val_loss - self.early_stopping_delta:
            self.model.save_model_state_dict()
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement = self.epochs_without_improvement + 1

        # Early stopping condition
        if self.epochs_without_improvement >= self.early_stopping_patience:
            condition_match = True

        return condition_match


    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Logging and call early stopping condition

        args:
            val_loss: validation loss
            epoch: current epoch

        returns:
            condition_match: True if has to stop training False if doesn't
        """
        condition_match = self.__check_early_stopping(val_loss)

        self.logger.info('Epoch: {} \tEarlyStopping: {} out of {}. Val loss: {:.6f}'.format(
            epoch, self.epochs_without_improvement, self.early_stopping_patience, val_loss))

        if condition_match:
            self.logger.info(
                f"Early stopping! Validation loss hasn't improved for {self.early_stopping_patience} epochs")
            self.model.load_model_state_dict()
        
        return condition_match
