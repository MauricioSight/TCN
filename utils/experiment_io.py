import json
import pickle
import torch
import yaml
from datetime import datetime
from pathlib import Path

def search_tune_id(train_config, tune_config):
    base_runs_dir = Path("runs")
    
    if not base_runs_dir.exists() or not base_runs_dir.is_dir():
        print(f"Runs directory not found at {base_runs_dir}")
        return None
    
    for run_dir in base_runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        train_config_path = run_dir / "config.yaml"
        tune_config_path = run_dir / "tune_config.yaml"
        
        if not train_config_path.exists() or not tune_config_path.exists():
            continue
        
        try:
            with open(train_config_path, "r") as f:
                run_train_config = yaml.safe_load(f) or {}
            with open(tune_config_path, "r") as f:
                run_tune_config = yaml.safe_load(f) or {}
            
            tune_id = run_tune_config['run_id']

            # Remove 'run_id' key if present
            run_train_config.pop("run_id", None)
            run_tune_config.pop("run_id", None)
            
            if run_train_config == train_config and run_tune_config == tune_config:
                return tune_id
        except Exception as e:
            print(f"Error processing configs in {run_dir}: {e}")
    
    return None


def search_run_id(config, file_name):
    # Get the base runs directory
    base_runs_dir = Path("runs")
    
    if not base_runs_dir.exists() or not base_runs_dir.is_dir():
        print(f"Runs directory not found at {base_runs_dir}")
        return None
    
    # Walk through each run directory
    for run_dir in base_runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        run_id = run_dir.name
        config_path = run_dir / f"{file_name}.yaml"
        
        if not config_path.exists():
            continue
            
        # Load the config file
        try:
            with open(config_path, "r") as f:
                run_config = yaml.safe_load(f)
                
            # Create a copy of run_config without the run_id for comparison
            comparison_config = run_config.copy()
            if "run_id" in comparison_config:
                comparison_config.pop("run_id")
                
            # Compare configs
            if comparison_config == config:
                return run_id
        except Exception as e:
            print(f"Error processing {config_path}: {e}")
    
    return None

def generate_run_id(keys) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = "_".join(keys) + f"_{timestamp}"
    return run_id

def get_run_id(config: dict, keys, default_file_name: str = 'config') -> str:
    run_id = search_run_id(config, default_file_name)

    if run_id is None:
        run_id = generate_run_id(keys)
    
    return run_id

def get_tune_id(tune_config: dict, train_config: dict, keys) -> str:
    tune_id = search_tune_id(train_config, tune_config)

    if tune_id is None:
        tune_id = generate_run_id(keys)
    
    return tune_id
    

def get_run_dir(run_id: str, base_dir: str = "runs") -> Path:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_run_artifacts(run_dir, config, y_true = None, y_scores = None, metrics = None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / f"config.yaml", "w") as f:
        yaml.dump(config, f)

    # Save y_true and predictions
    if y_true is not None and y_scores is not None:
        torch.save({'y_true': y_true, 'y_scores': y_scores}, 
                   run_dir / f"{config['phase']}_labels_predictions.pt", pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # Save metrics
    if metrics is not None:
        with open(run_dir / f"{config['phase']}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

def load_metrics(run_dir: str, phase: str) -> dict:
    run_dir = Path(run_dir)
    metrics_path = run_dir / f"{phase}_metrics.json"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    return metrics

def save_run_tune(run_dir, tune_config: dict, train_config: dict):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "tune_config.yaml", "w") as f:
        yaml.dump(tune_config, f)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(train_config, f)
