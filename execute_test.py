import logging

from logger.base import Logger
from tracker.wandb_tracker import WandBTracker
from utils.experiment_io import get_run_dir, save_run_artifacts
from utils.config_handle import load_config, flatten_dict
from utils.get_device import get_device
from utils.seed_all import seed_all

from data_loader.factory import DataLoaderFactory
from data_pre_processing.factory import DataPreProcessingFactory

from modeling.structure.factory import ModelingStructureFactory
from modeling.inference.factory import ModelingInferenceFactory

from metrics.factory import MetricsFactory

def main(config=None, X=None, y_true=None):
    """
    Perform testing phase

    args:
        config: optional if what to use specific config
        X: optional if what to use specific data
        y_true: optional if what to use specific data

    returns:
        metrics
    """

    if config is None:
        config = load_config(run_id="cnn_MNIST_20250829_134355")

    if 'run_id' not in config:
        raise ValueError("Missing run id in config")
    
    config['phase'] = 'test' # persist testing phase
    run_id = config['run_id']
    run_dir = get_run_dir(run_id)

    save_run_artifacts(run_dir, config)

    # Setup logger
    logger = Logger(name="testing", log_file=f"{run_dir}/test_output.log", 
                    level=logging.DEBUG if 'debug' in config and config['debug'] else logging.INFO)

    # log run id
    logger.info("Initiating testing...")
    logger.info(f"[ RUN ID: {run_id} ]")

    seed = 0
    seed_all(seed)
    logger.debug(f"[ Using Seed : {seed} ]")

    # 1. Load the dataset
    if X is None or y_true is None:
        logger.debug("Loading data...")
        dataset_loader = DataLoaderFactory().get(config, logger)
        pre_processing = DataPreProcessingFactory().get(config, logger)
        X, y_true = pre_processing.initialize(dataset_loader)
        logger.info("Data loaded successfully.")
    else:
        logger.info("Using provided data for training and validation.")

    # 3. Initializations
    logger.debug("Initializing components...")

    device = get_device()
    logger.info(f"Using device: {device}")

    # 3.1 Model
    logger.debug("Initializing model...")
    model = ModelingStructureFactory().get(config, logger, device)
    model.compile()

    # 3.2 Tracker
    logger.debug("Initializing tracker...")
    flat_config = flatten_dict(config)
    tracker = WandBTracker(config=flat_config, run_name=run_id, model=model)

    # 3.3 Inference
    logger.debug("Initializing inference...")
    model_inference = ModelingInferenceFactory().get(config, logger, device)

    logger.debug("Initializing metrics...")
    metrics_handler = MetricsFactory().get(config, logger)
    
    # 4. Test
    logger.debug("Starting testing...")
    y_true_val, y_scores, val_loss = model_inference.inference(model, X, y_true)
    logger.info(f"Testing completed. Testing loss: {val_loss}")

    # 5. Get metrics
    logger.debug("Getting metrics...")
    metrics = metrics_handler.get_overall_metrics(y_true_val, y_scores)
    tracker.log_metrics({**metrics, 'test_loss': val_loss})

    logger.info("Execution completed.")

    # 6. Finish the tracker
    tracker.finish()

    # 7. Save run artifacts
    logger.debug("Saving run artifacts...")
    save_run_artifacts(run_dir, config, y_true=y_true_val, y_scores=y_scores, metrics=metrics)
    logger.info("Run artifacts saved.")
    
    logger.info("Testing completed.")

    return metrics

if __name__ == "__main__":
    main()
