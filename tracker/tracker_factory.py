from tracker.wandb_tracker import WandBTracker
from tracker.mlflow_tracker import MLflowTracker

def get_tracker(tracker_name: str):
    if tracker_name == "wandb":
        return WandBTracker()
    elif tracker_name == "mlflow":
        return MLflowTracker()
    else:
        raise ValueError(f"Unsupported tracker: {tracker_name}")
