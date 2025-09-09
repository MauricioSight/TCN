import logging
import optuna
from optimizer.base_optimizer import BaseOptimizer
from utils.experiment_io import get_run_dir

class OptunaOptimizer(BaseOptimizer):
    def __init__(self, config):
        self.run_id = config.get("run_id")
        self.run_dir = get_run_dir(self.run_id)
        self.sampler_name = config.get("algorithm")
        self.search_space = config.get("search_space")
        self.direction = config.get("direction", "maximize")
        self.n_trials = config.get("n_trials", 20)

        optuna_logger = optuna.logging.get_logger("optuna")
        handler = logging.FileHandler(f"{self.run_dir}/optuna.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        optuna_logger.addHandler(handler)
        

    def _get_sampler(self):
        if self.sampler_name == "tpe":
            return optuna.samplers.TPESampler()
        elif self.sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler()
        elif self.sampler_name == "random":
            return optuna.samplers.RandomSampler()
        else:
            raise ValueError(f"Unknown Optuna sampler: {self.sampler_name}")

    def _suggest(self, trial, name, space):
        kind = space[0]
        args = space[1:]
        step = None
        if len(args) > 2 and kind != "categorical":
            step = args[-1]
            args = args[:-1]

        if kind == "float":
            return trial.suggest_float(name, *[float(a) for a in args], step=step)
        elif kind == "float_log":
            return trial.suggest_float(name, *[float(a) for a in args], step=step, log=True)
        elif kind == "int":
            return trial.suggest_int(name, *[int(a) for a in args])
        elif kind == "categorical":
            return trial.suggest_categorical(name, args)
        elif kind == "discrete_uniform":
            return trial.suggest_float(name, *[float(a) for a in args], step=step)
        else:
            raise ValueError(f"Unknown suggestion type: {kind}")

    def optimize(self, objective_func):
        def wrapped_objective(trial):
            params = {
                k: self._suggest(trial, k, v)
                for k, v in self.search_space.items()
            }
            return objective_func(params)

        storage = self.run_dir / "optuna.db"
        storage = "sqlite:///" + str(storage)
        study = optuna.create_study(study_name=self.run_id, direction=self.direction, sampler=self._get_sampler(), 
                                    storage=storage, load_if_exists=True)

        study.optimize(wrapped_objective, n_trials=self.n_trials)
        return study.best_trial
