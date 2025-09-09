import logging

import numpy as np

from logger.base import Logger
from metrics.factory import MetricsFactory
from utils.experiment_io import get_run_dir, save_run_artifacts
from utils.config_handle import load_config
from utils.seed_all import seed_all

def main(phase='train', run_id: str ='cnn_MNIST_20250829_134355', y_true=None, y_scores=None):
    """
    Get metrics from previous run_id

    args:
        phase: train or test
        run_id
        y_true: optional if what to get metric with specif data
        y_scores: optional if what to get metric with specif data

    returns:
        metrics
    """

    if config is None:
        config = load_config(run_id=run_id)

    if 'run_id' not in config:
        raise ValueError("Missing run id in config")
    
    config['phase'] = phase
    run_id = config['run_id']
    run_dir = get_run_dir(run_id)

    save_run_artifacts(run_dir, config)

    # Setup logger
    logger = Logger(name="get_metrics", log_file=f"{run_dir}/metrics_output.log", 
                    level=logging.DEBUG if 'debug' in config and config['debug'] else logging.INFO)

    # log run id
    logger.info("Initiating get metrics...")
    logger.info(f"[ RUN ID: {run_id} ]")

    seed = 0
    seed_all(seed)
    logger.debug(f"[ Using Seed : {seed} ]")

    # 1. Load the dataset
    if y_true is None or y_scores is None:
        logger.debug("Loading data...")
        data = np.load(run_dir / f"{config['phase']}_labels_predictions.npz")
        y_true = data["y_true"]
        y_scores = data["y_scores"]
        logger.info("Data loaded successfully.")
    else:
        logger.info("Using provided data for training and validation.")

    # 3.3 Trainer
    logger.debug("Initializing metrics...")
    metrics_handler = MetricsFactory().get(config, logger)

    # 6. Get metrics
    logger.debug("Getting metrics...")
    metrics = metrics_handler.get_overall_metrics(y_true, y_scores)

    logger.info("Execution completed.")

    return metrics

if __name__ == "__main__":
    main()
