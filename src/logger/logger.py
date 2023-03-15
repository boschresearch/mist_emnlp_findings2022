# This source code is from the STEPS Parser (w/ adaptations by Sophie Henning)
#   (https://github.com/boschresearch/steps-parser/blob/master/src/logger/logger.py)
# Copyright (c) 2020 Robert Bosch GmbH
# This source code is licensed under the AGPL v3 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
# Author: Stefan Grünewald


import mlflow
import logging
import logging.config
import os
from typing import Dict, Optional

DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(message)s"},
        "datetime": {"format": "%(asctime)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
            },
        "debug_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "debug.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "datetime",
            "filename": "info.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": [
            "console",
            "debug_file_handler",
            "info_file_handler"
        ]
    }
}


class Logger:
    """Class for logging messages, metrics, and artifacts during training. May use MLFlow for storing experiment data.
    """
    def __init__(self, save_dir, verbosity=logging.DEBUG, use_mlflow=False, experiment_id=None, run_name=None,
                 start_mlflow_run=True, mlruns_folder: Optional[str] = None):
        """
        Args:
            save_dir: Directory to save text log to.
            verbosity: Verbosity of text logging.
            use_mlflow: Whether to use MLFlow in addition to text logging (default: False).
            experiment_id: Experiment ID for MLFlow.
            run_name: Run name for MLFlow.
        """
        for _, handler in DEFAULT_CONFIG['handlers'].items():
            if 'filename' in handler and save_dir is not None:
                handler['filename'] = os.path.join(save_dir, handler['filename'])

        logging.config.dictConfig(DEFAULT_CONFIG)

        self.text_logger = logging.getLogger()
        self.text_logger.setLevel(verbosity)

        self.use_mlflow = use_mlflow
        if self.use_mlflow and start_mlflow_run:
            if mlruns_folder:
                mlflow.set_tracking_uri(f"file://{mlruns_folder}")
            mlflow.set_experiment(experiment_id)
            mlflow.start_run(run_name=run_name)

    def info(self, msg):
        """Log message with level INFO with the text logger."""
        self.text_logger.info(msg)

    def debug(self, msg):
        """Log message with level DEBUG with the text logger."""
        self.text_logger.debug(msg)

    def warning(self, msg):
        """Log message with level WARNING with the text logger."""
        self.text_logger.warning(msg)

    def log_metric(self, metric_name, value, percent=True, step=None):
        """Log a training/evaluation metric.

        Args:
            metric_name: Name of the metric to log.
            value: Value of the metric.
            percent: Whether to log the metric as a percentage in the text log (default: True).
            step: Epoch to log the metric for.
        """
        value = float(value)
        if percent:
            self.info("{}: {:.2f}%".format(metric_name, value*100))
        else:
            self.info("{}: {:.4}".format(metric_name, value))

        if self.use_mlflow:
            mlflow.log_metric(metric_name, value, step=step)

    def log_param(self, param_name, value):
        """Log a parameter."""
        raise NotImplementedError("log_param not implemented yet.")

    def log_artifact(self, local_path, artifact_path=None):
        """Log an artifact. Calling this method only has an effect when `use_mlflow` is set to True.

        Args:
            local_path: Path (in MLFlow directory) to log the artifact under.
            artifact_path: If provided, the directory in `artifact_uri` to write to.
        """
        if not self.use_mlflow:
            self.text_logger.info(f"Ignoring request to log artifact {local_path} because use_mlflow was set to False.")
        else:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_config(self, config):
        """Log a config.

        Args:
            config: Nested dictionary of parameters.
        """
        flat_config = dict()
        _flatten_dict(config, flat_config)

        for param, value in flat_config.items():
            self.text_logger.info(f"{param} = {value}")

        if self.use_mlflow:
            mlflow.log_params(flat_config)

    def log_epoch_metrics(self, metrics: Dict[str, float], step: int = None, suffix: str = ""):
        """Log metrics for one epoch.

        Args:
            metrics: Metrics to log.
            step: Epoch to log the metrics for.
            suffix: Suffix to add to metric names (e.g. "_train").
        """
        for metric_name, metric_value in metrics.items():
            if "Accuracy" in metric_name or "F1" in metric_name:
                self.log_metric(metric_name, metric_value, percent=True, step=step)
            else:
                self.log_metric(metric_name, metric_value, percent=False, step=step)
        # # Log loss
        # self.log_metric(f"loss{suffix}", metrics["loss"], percent=False, step=step)
        #
        # # Log accuracy if evaluated
        # if "accuracy" in metrics.keys():
        #     self.log_metric(f"accuracy{suffix}", metrics["accuracy"], percent=True, step=step)

    def log_final_metrics_basic(self, metrics, suffix=""):
        """Log the final evaluation metrics (as returned by the conll18_ud_eval.py script).

        Args:
            metrics: Evaluation metrics as returned by the conll18_ud_eval.py script.
            suffix: Suffix to add to metric names (e.g. "_test").
        """
        self.log_metric("uas_final" + suffix, metrics["UAS"].f1, percent=True)
        self.log_metric("las_final" + suffix, metrics["LAS"].f1, percent=True)

    def log_final_metrics_enhanced(self, metrics, suffix=""):
        """Log the final evaluation metrics (as returned by the iwpt20_xud_eval.py script).

        Args:
            metrics: Evaluation metrics as returned by the iwpt20_xud_eval.py script.
            suffix: Suffix to add to metric names (e.g. "_test").
        """
        self.log_metric("euas_final" + suffix, metrics["EUAS"].f1, percent=True)
        self.log_metric("eulas_final" + suffix, metrics["EULAS"].f1, percent=True)
        self.log_metric("elas_final" + suffix, metrics["ELAS"].f1, percent=True)

    def log_final_metrics_ner(self, metrics, suffix=""):
        """Log the final evaluation metrics (as returned by the ner_eval.py script).

        Args:
            metrics: Evaluation metrics as returned by the ner_eval.py script.
            suffix: Suffix to add to metric names (e.g. "_test").
        """
        self.log_metric("precision_final" + suffix, metrics["Micro"].precision, percent=True)
        self.log_metric("recall_final" + suffix, metrics["Micro"].recall, percent=True)
        self.log_metric("f1_final" + suffix, metrics["Micro"].f1, percent=True)


def _flatten_dict(input_dict, output_dict, prefix="", delimiter="."):
    """Flatten the nested dictionary input_dict, writing to output_dict. """
    for key, value in input_dict.items():
        if not isinstance(value, dict):
            output_dict[prefix+key] = value
        else:
            _flatten_dict(value, output_dict, prefix=prefix+key+delimiter, delimiter=delimiter)
