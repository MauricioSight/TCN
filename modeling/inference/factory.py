from torch import device as TorchDevice

from modeling.inference.pytorch_base import PytorchInference
from logger.base import Logger

class ModelingInferenceFactory:
    """
    Base class for inference factory.
    """
    def get(self, config: dict, logger: Logger, device: TorchDevice) -> PytorchInference:
        name = config.get('modeling', {}).get('inference', {}).get('name')

        if name == 'dnn_class':
            from modeling.inference.dnn_class_inference import DNNClassInference

            return DNNClassInference(config, logger, device)

        if name == 'pred':
            from modeling.inference.pred_inference import PredInference

            return PredInference(config, logger, device)

        else:
            raise ValueError(
                f"Unsupported ModelingInferenceFactory name: {name}")