from metrics.base import InferenceMetrics
from logger.base import Logger

class MetricsFactory:
    """
    Base class for metrics factory.
    """
    def get(self, config: dict, logger: Logger) -> InferenceMetrics:
        name = config.get('metrics', {}).get('name')

        if name == 'multi_class':
            from metrics.multi_class import MultiClassificationMetrics

            return MultiClassificationMetrics(config, logger)

        else:
            raise ValueError(
                f"Unsupported MetricsFactory name: {name}")