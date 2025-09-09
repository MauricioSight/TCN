import wandb
from tracker.base_tracker import BaseTracker

class WandBTracker(BaseTracker):
    def __init__(self, run_name: str, config: dict, model):
        super().__init__()
        self.start_run(run_name=run_name, config=config, model=model)

    def start_run(self, run_name: str, config: dict, model):
        wandb.init(project="w-sizes-fixed_params", name=run_name, config=config)
        
        wandb.watch(model, log_freq=100)

    def log_metrics(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)

    def log_artifact(self, filepath: str):
        wandb.save(filepath)

    def finish(self):
        wandb.finish()
