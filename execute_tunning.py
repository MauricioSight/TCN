import json
import logging

from logger.base import Logger
from optimizer.factory import OptimizerFactory
from utils.experiment_io import get_run_id, get_run_dir, get_tune_id, save_run_tune
from execute_train_validation import main as execute_train_validation_main
from utils.config_handle import load_config

def apply_trial_to_config(config: dict, trial_params: dict) -> dict:
    new_config = config.copy()

    for flat_key, value in trial_params.items():
        keys = flat_key.split("-")
        current = new_config

        # Navigate to the parent dict
        for k in keys[:-1]:
            current = current.setdefault(k, {})

        # Overwrite the final key
        current[keys[-1]] = value

    return new_config

def main(tune_run_id: str = None, config_file='tune-mlp'):
    # Load the configuration
    tune_config = load_config(default_file_name=config_file, run_id=tune_run_id)
    train_config = load_config(config_name=tune_config['train_config'], default_file_name='train_config')

    if 'run_id' not in tune_config:
        run_id = get_tune_id(tune_config, train_config, [tune_config['framework'], tune_config['algorithm'], 
                                                         tune_config['train_config']])
        tune_config['run_id'] = run_id
    
    run_id = tune_config['run_id']
    run_dir = get_run_dir(run_id)

    save_run_tune(run_dir, tune_config=tune_config, train_config=train_config)

    # Setup logger
    tune_logger = Logger(name="tune", log_file=f"{run_dir}/tune_output.log", 
                    level=logging.DEBUG if 'debug' in tune_config and tune_config['debug'] else logging.INFO)

    tune_logger.info("Starting tuning process...")
    tune_logger.info(f"[ RUN ID: {run_id} ]")

    # Initialize the optimizer
    tune_logger.info("Initializing optimizer...")

    objective_metric = tune_config['objective_metric']
 
    def objective_fn(params):
        tune_logger.info(f"Trial parameters: ")
        tune_logger.info(json.dumps(params, indent=4))

        updated_config = apply_trial_to_config(train_config, params)
        tune_logger.debug(f"Updated config: {json.dumps(updated_config, indent=4)}")

        # Set the run ID for this trial
        experiment_id = get_run_id(updated_config, [updated_config['modeling']['structure']['name'], updated_config['data_loader']['name']])
        updated_config['run_id'] = experiment_id
        
        metrics = execute_train_validation_main(config=updated_config)

        tune_logger.info(f"Objective metric: {metrics[objective_metric]}")
        return metrics[objective_metric]
    
    optimizer_factory = OptimizerFactory(tune_config)
    optimizer = optimizer_factory.get_optimizer()
    optimizer.optimize(objective_fn)

    tune_logger.info("Tuning completed.")


if __name__ == "__main__":
    main()
