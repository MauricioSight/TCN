from metrics.base import InferenceMetrics
from logger.base import Logger

class MetricsFactory:
    """
    Base class for metrics factory.
    """
    def get(self, config: dict, logger: Logger, context={}) -> InferenceMetrics:
        name = config.get('metrics', {}).get('name')

        if name == 'multi_class':
            from metrics.multi_class import MultiClassificationMetrics

            return MultiClassificationMetrics(config, logger, context)

        if name == 'anomaly_detector':
            from metrics.ad_metrics import AnomalyDetectorMetrics

            return AnomalyDetectorMetrics(config, logger, context)

        if name == 'natasha_ae':
            from metrics.nat_ae_metrics import NatashaAEMetrics

            return NatashaAEMetrics(config, logger, context)

        else:
            raise ValueError(
                f"Unsupported MetricsFactory name: {name}")