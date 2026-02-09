import logging

import pandas as pd
import torch
import numpy as np

from logger.base import Logger
from metrics.factory import MetricsFactory
from modeling.inference.factory import ModelingInferenceFactory
from modeling.structure.factory import ModelingStructureFactory
from utils.experiment_io import get_run_dir, save_run_artifacts
from utils.config_handle import load_config
from utils.get_device import get_device
from utils.seed_all import seed_all

def main(phase='test', run_id: str ='aero_AEID_20260128_110602', y_true=None, y_scores=None):
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
        cache = torch.load(run_dir / f"{config['phase']}_labels_predictions.pt", weights_only=False)
        y_true, y_scores = cache['y_true'], cache['y_scores']
        logger.info("Data loaded successfully.")
    else:
        logger.info("Using provided data for training and validation.")

    device = get_device()
    logger.info(f"Using device: {device}")
    
    # 3.1 Model
    logger.debug("Initializing model...")
    model = ModelingStructureFactory().get(config, logger, device)
    model.compile()

    logger.debug("Initializing inference...")
    model_inference = ModelingInferenceFactory().get(config, logger, device)

    # 3.3 Trainer
    logger.debug("Initializing metrics...")
    train_data = np.float32(np.random.random((1, 1, 18450))) # .rand((1, 1, 64, 58), dtype=torch.float32)
    context={'model': model, 'model_inference': model_inference, 'train_data': train_data}
    metrics_handler = MetricsFactory().get(config, logger, context)

    # 6. Get metrics
    logger.debug("Getting metrics...")
    threshold = pd.read_json(f'{run_dir}/train_metrics.json')['optimal_threshold'].values[0]
    metrics = metrics_handler.get_overall_metrics(y_true, y_scores, threshold=threshold)

    logger.info("Execution completed.")

    return metrics

if __name__ == "__main__":
    main()
