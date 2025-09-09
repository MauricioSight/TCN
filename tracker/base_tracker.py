class BaseTracker:
    def start_run(self, run_name: str, config: dict):
        pass

    def log_metrics(self, metrics: dict, step: int):
        pass

    def log_artifact(self, filepath: str):
        pass  # Optional

    def finish(self):
        pass
