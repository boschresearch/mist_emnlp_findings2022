# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Sophie Henning

"""Run an evaluation configuration"""

import mlflow
import argparse
import numpy as np
from pathlib import Path

from cross_validation import run_cross_validation
from config.config_utils import read_json
from util.mlflow_utils import create_mlflow_experiment
from util.util import prepend_optional_path

parser = argparse.ArgumentParser()
parser.add_argument("evaluation_config", type=str, help="Path to evaluation config")
parser.add_argument("--repo_path", type=str,
                    help="Optional: path to repo, will be prepended to all paths in config files.")
args = parser.parse_args()

config = read_json(args.evaluation_config)

repo_path = None if args.repo_path is None else Path(args.repo_path)

task2dataset2domains_ignored_in_metrics = None
if "domains_ignored_in_metrics" in config:
    task2dataset2domains_ignored_in_metrics = config["domains_ignored_in_metrics"]

experiment_id = create_mlflow_experiment(prepend_optional_path(Path(config["mlruns_folder"]), repo_path),
                                         config["name"])

with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.log_artifact(args.evaluation_config)
    cv_cfg_path = config["cv_config"]
    best_model_cfg_path = config["best_model_config"]

    # Run CV here returns a dict from split number to a dict with test metrics
    test_results, tasks = run_cross_validation(cv_cfg_path, best_model_cfg_path, dict(), experiment_id=experiment_id,
                                               evaluate=True,
                                               task2dataset2domains_ignored_in_metrics=task2dataset2domains_ignored_in_metrics,
                                               repo_path=repo_path)
    splits = sorted(test_results.keys())

    logged_metrics = {metric: [] for metric in test_results[splits[0]].keys()}

    # Gather the metrics across splits and log them
    for split in splits:
        split_metrics = test_results[split]
        for metric, value in split_metrics.items():
            if metric in logged_metrics.keys():
                logged_metrics[metric].append(value)
                mlflow.log_metrics({f"{metric}_fold{split}": value})

    # Compute mean and standard deviation
    for metric, values in logged_metrics.items():
        mlflow.log_metrics({f"cv_{metric}": np.mean(values).item(),
                            f"cv_{metric}_stdev": np.std(values)})