from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def optimize(self, objective_func, search_space: dict, n_trials: int = 20):
        pass
