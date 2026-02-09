from data_loader.base import DataLoader
from logger.base import Logger

class DataLoaderFactory:
    """
    Base class for data_loader factory.
    """
    def get(self, config: dict, logger: Logger) -> DataLoader:
        name = config.get('data_loader', {}).get('name')

        if name == 'MNIST':
            from data_loader.MNIST_loader import MNISTLoader

            return MNISTLoader(config, logger)
        
        if name == 'TOWIDS':
            from data_loader.TOWIDS_loader import TOWIDSLoader

            return TOWIDSLoader(config, logger)
        
        if name == 'AEID':
            from data_loader.AEID_loader import AEIDLoader

            return AEIDLoader(config, logger)
        
        else:
            raise ValueError(
                f"Unsupported DataLoaderFactory name: {name}")
