import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from Condition2Cure.utils.helpers import create_directories
from Condition2Cure.entities.config_entity import DriftMonitoringConfig
from Condition2Cure import logger


class DriftMonitor:
    def __init__(self, config: DriftMonitoringConfig):
        self.config = config

    def monitoring(self):
        logger.info("Running drift monitoring...")

        ref_path = Path(self.config.reference_data)
        curr_path = Path(self.config.current_data)
        output_dir = Path(self.config.output_dir)

        # Load data
        reference_data = pd.read_csv(ref_path)
        current_data = pd.read_csv(curr_path)

        # Generate Evidently Report
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)

        # Save reports
        create_directories([output_dir])

        report_path_html = output_dir / "drift_report.html"
        report_path_json = output_dir / "drift_report.json"

        report.save_html(str(report_path_html))
        report.save_json(str(report_path_json))


        logger.info(f"Drift report saved to: {report_path_html}")
        logger.info(f"Drift report (JSON) saved to: {report_path_json}")