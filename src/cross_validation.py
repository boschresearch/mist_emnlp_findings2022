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

import numpy as np
import mlflow
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

from config.config_parser import ConfigParser
from config.config_utils import read_json
from train import init_config_modification, main as train_main
from util.mlflow_utils import create_mlflow_experiment
from util.util import prepend_optional_path

from typing import Dict, Union, Optional, Tuple, List


def run_cross_validation(cv_config_path: Path, base_model_path: Path, params: Dict[str, str],
                         epochs: Optional[int] = None, experiment_id: Optional[str] = None, store_best: bool = True,
                         evaluate: bool = False, run: Optional[Union[int, str]] = None,
                         keys2value_providing_keys: Optional[Dict[str, str]] = None,
                         task2dataset2domains_ignored_in_metrics: Optional[Dict[str, Dict[str, List[str]]]] = None,
                         repo_path: Optional[Path] = None) \
        -> Union[Tuple[float, str], Tuple[Dict, List[str]]]:
    """

    Args:

        cv_config_path: path to a cross-validation config JSON file
        base_model_path: path to model config to use
        params: parameters of the model to change from the config under base_model_path;
            dict keys must be valid paths in the base model config
        epochs: number of epoch for which every individual training will be run
        experiment_id: ID of MLFlow experiment as part of which the cross-validation is run
        store_best: If set to True, k models will be stored (each being the best of a single of the k trainings).
        evaluate: Evaluate CV models on test set.
        run: optional integer to mark different CV runs (e.g., during hyperparameter search)
        keys2value_providing_keys: dictionary from parameter keys (in model config) to keys in params (argument of this function) whose value they should take
        task2dataset2domains_ignored_in_metrics: specify domains to ignore for task-dataset combinations
            (cf. dataset2domains_ignored_in_metrics in ModalTask)
        repo_path: Path to repo, will be prepended to all paths to config files (including cv_config_path and base_model_path)

    Returns:
        average cross-validation validation metric or a dictionary containing test set results
    """
    cv_config_path = prepend_optional_path(cv_config_path, repo_path)
    base_model_path = prepend_optional_path(base_model_path, repo_path)

    cv_config = read_json(cv_config_path)

    base_model_config = read_json(base_model_path)

    run_name = f"CV_{run}" if run else "CV"

    if experiment_id is None:
        nested_mlflow_run = False
        # Set MLFlow tracking URI to avoid storing data in home directory
        if evaluate:
            # Use base model config's data for storing if evaluating on test set
            mlruns_folder = base_model_config["mlruns_folder"]
            config_name = base_model_config["name"]
        else:
            # Use CV config's data for storing if not evaluating on test set
            mlruns_folder = cv_config["mlruns_folder"]
            config_name = cv_config["name"]

        mlruns_folder = prepend_optional_path(mlruns_folder, repo_path)

        # Create an experiment for nicer UI
        experiment_id = create_mlflow_experiment(mlruns_folder, config_name)
    else:
        nested_mlflow_run = True

    with mlflow.start_run(nested=nested_mlflow_run, experiment_id=experiment_id, run_name=run_name) as outer:
        mlflow.log_artifact(cv_config_path)
        mlflow.log_params(params)

        if evaluate:
            # Track test set metrics
            test_results = {}

        # Track validation metrics
        val_metrics = []
        val_losses = []
        other_metrics = defaultdict(list)

        # store metric name
        val_metric_name = None

        # Set hyperparameters that are shared across all splits
        mod_list = [f"{key}={json.dumps(value)}" for key, value in params.items()]

        # Check if number of epochs has been specified
        if epochs is not None:
            mod_list.append(f"trainer.max_epochs={epochs}")
            mod_list.append(f"trainer.save_period={epochs + 1}") # Don't store intermediate models

        # Check if some values should be copied over from 'params' argument
        if keys2value_providing_keys is not None:
            # Set keys to values of other keys they are supposed to take
            for key, value_providing_key in keys2value_providing_keys.items():
                mod_list.append(f"{key}={json.dumps(params[value_providing_key])}")

        # Check if some domains should be ignored in metrics (for cross-domain experiment)
        if task2dataset2domains_ignored_in_metrics is not None:
            mod_list.append(f"domains_ignored_in_metrics={json.dumps(task2dataset2domains_ignored_in_metrics)}")

        # Paths for data loaders
        train_config_path = "data_loader.paths.train"
        val_config_path = "data_loader.paths.val"
        # Append path to test set if in config
        # If test set should be different for different CV configs, this should be specified in the respective CV config
        test_config_path = "data_loader.paths.test"
        if "test" in cv_config:
            for task_name, test_data_path in cv_config['test'].items():
                test_data_path = prepend_optional_path(test_data_path, repo_path)
                mod_list.append(f"{test_config_path}.{task_name}=\"{test_data_path}\"")

        # Get information on CV split / CV configurations
        splits = cv_config["splits"]

        # cv_config and all sub-dicts are OrderedDicts, so iteration over keys is always in the same order
        for split_number in splits.keys():
            with mlflow.start_run(nested=True, experiment_id=experiment_id,
                                  run_name=f"{run_name}_fold_{split_number}") as inner:
                # Start run for i-th fold
                mod_list_split = mod_list.copy()  # Copy list for each split (different train and val paths)

                # Append paths to train/val set and also test set (if applicable) for this CV config
                split_info = splits[split_number]
                for task_name, train_data_path in split_info['train'].items():
                    train_data_path = prepend_optional_path(train_data_path, repo_path)
                    mod_list_split.append(f"{train_config_path}.{task_name}=\"{train_data_path}\"")
                    if task_name in split_info['val']:
                        val_data_path = split_info['val'][task_name]
                        val_data_path = prepend_optional_path(val_data_path, repo_path)
                        mod_list_split.append(f"{val_config_path}.{task_name}=\"{val_data_path}\"")
                    if 'test' in split_info:
                        # Different test sets per CV config
                        # This also overrides a test set previously defined for all CV configs
                        if task_name in split_info['test']:
                            test_data_path = split_info['test'][task_name]
                            test_data_path = prepend_optional_path(test_data_path, repo_path)
                            mod_list_split.append(f"{test_config_path}.{task_name}=\"{test_data_path}\"")

                modification = init_config_modification(mod_list_split)
                config_parser = ConfigParser(base_model_config, modification=modification, use_mlflow=True,
                                             start_mlflow_run=False, store_best=store_best, repo_path=repo_path)

                # Random seed is set at the beginning of train_main, directly before model and data loader
                # initialization

                # Call train_main function as a subprocess to ensure that memory is freed
                result_queue = mp.Queue()
                p = mp.Process(target=train_main, args=(config_parser, evaluate, result_queue))
                p.start()
                p.join()
                result = result_queue.get()

                if evaluate:
                    test_results[split_number] = result.test_results
                    tasks = result.tasks

                best_val_metric = result.best_val_metric
                loss_of_best_model = result.val_loss_of_best_model

                mlflow.log_metrics({f"best_{result.val_metric}_val": best_val_metric,
                                    "best_loss_val": loss_of_best_model})

                val_metrics.append(best_val_metric)
                val_losses.append(loss_of_best_model)
                for key, value in result.other_metrics_of_best_model.items():
                    mlflow.log_metrics({f"best_{key}": value})
                    other_metrics[key].append(value)
                val_metric_name = result.val_metric

        # Log the metrics from this CV run - also do this if evaluating for sanity checks
        cv_metric = np.mean(val_metrics)
        cv_loss = np.mean(val_losses)

        mlflow.log_metrics({f"cv_{val_metric_name}": cv_metric, "cv_loss": cv_loss})
        mlflow.log_metrics({f"cv_{key}": np.mean(value_list) for key, value_list in other_metrics.items()})

        if evaluate:
            return test_results, tasks
        else:
            return float(cv_metric), val_metric_name
