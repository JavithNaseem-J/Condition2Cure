from Condition2Cure.config.config import ConfigurationManager
from Condition2Cure.components.drift_monitoring import DriftMonitor


class DriftMonitoringPipeline:
    def __init(self):
        pass

    def run(self):
        config = ConfigurationManager()
        drift_monitoring_config = config.get_drift_monitoring_config()
        monitor = DriftMonitor(config=drift_monitoring_config)
        monitor.monitoring()