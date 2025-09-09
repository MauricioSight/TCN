from optimizer.base_optimizer import BaseOptimizer

from optimizer.optuna_optimizer import OptunaOptimizer

class OptimizerFactory:
    def __init__(self, config):
        self.config = config
        self.optimizer = self.__create_optimizer()

    def __create_optimizer(self):
        framework = self.config.get("framework")

        if framework == "optuna":
            return OptunaOptimizer(self.config)
        else:
            raise ValueError(f"Unsupported optimizer: {framework}")
    
    def get_optimizer(self) -> BaseOptimizer:
        return self.optimizer
    
