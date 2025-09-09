from torch import device as TorchDevice

from modeling.structure.pytorch_base import PytorchModelStructure
from logger.base import Logger

class ModelingStructureFactory:
    """
    Base class for model structure factory.
    """
    def get(self, config: dict, logger: Logger, device: TorchDevice) -> PytorchModelStructure:
        name = config.get('modeling', {}).get('structure', {}).get('name')

        if name == 'mlp':
            from modeling.structure.mlp import MLP

            return MLP(config, logger, device)
        
        if name == 'tcn_pred':
            from modeling.structure.tcn_pred import TCNPred

            return TCNPred(config, logger, device)
        
        if name == 'cae':
            from modeling.structure.cae import CAE

            return CAE(config, logger, device)
        
        if name == 'lstmae':
            from modeling.structure.lstmae import LSTMAE

            return LSTMAE(config, logger, device)

        else:
            raise ValueError(
                f"Unsupported ModelingStructureFactory name: {name}")