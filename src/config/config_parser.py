# This source code is from the PyTorch Template Project (w/ very heavy adaptations by Stefan Grünewald and Sophie Henning)
#   (https://github.com/victoresque/pytorch-template/blob/master/parse_config.py)
# Copyright (c) 2018 Victor Huang
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import transformers
from collections import OrderedDict

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from typing import OrderedDict as OrderedDictType  # Avoid confusion with collections.OrderedDict


import modules
import data_handling.data_loaders as data_loaders_module
import trainer.lr_schedules as lr_schedules_module

from trainer.trainer import Trainer
from trainer.lr_schedules import WarmupSchedule, SqrtSchedule  # needed for eval() to work in _init_lr_schedulers
from logger.logger import Logger
from data_handling.tasks import ModalTask, NO_DOMAINS
from data_handling.vocab import BasicVocab
from data_handling.sentence_dataset import SentenceDataset
from .config_utils import read_json, set_by_path, write_json
from util.util import prepend_optional_path


class ConfigParser:
    """
    This class parses the configuration JSON file and handles hyperparameters for training, checkpoint saving and logging.
    """

    def __init__(self, config: Dict, modification: Optional[Dict] = None, resume: Optional[str] = None,
                 run_id: Optional[str] = None, use_mlflow: bool = False, start_mlflow_run: bool = True,
                 store_best: bool = True, repo_path: Optional[Path] = None):
        """
        Args:
            config: Dict containing configuration/hyperparameters for training (contents of a `config.json` file)
            resume: String, path to a checkpoint to load. Default: None.
            run_id: Unique identifier for training processes. Used to save checkpoints and training log. Timestamp
              is being used as default.
            use_mlflow: Use mlflow for tracking the experiment
            start_mlflow_run: If using mlflow, start an mlflow run.
            store_best: Set this to False if best model should not be stored (e.g., in cross-validation).
            repo_path: Optional repo path to prepend to all paths found in configs.
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self.repo_path = repo_path

        experiment = self.config.get('experiment', None)
        run_name = self.config['name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S.%f')

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        save_dir = prepend_optional_path(save_dir, repo_path)

        if experiment is None:
            self._save_dir = save_dir / run_name / run_id
        else:
            self._save_dir = save_dir / experiment / run_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        self.store_best = store_best

        # Set up logging
        transformers.logging.set_verbosity_info()

        mlruns_folder = f"{config['mlruns_folder']}/{experiment}" if "mlruns_folder" in config.keys() else None
        mlruns_folder = prepend_optional_path(mlruns_folder, repo_path)
        self.logger = Logger(self.save_dir, use_mlflow=use_mlflow, experiment_id=experiment,
                             run_name=f"{run_name}{run_id}", start_mlflow_run=start_mlflow_run,
                             mlruns_folder=mlruns_folder)
        if self.save_dir is not None:
            self.logger.log_config(config)
            self.logger.log_artifact(self.save_dir / 'config.json')

    @classmethod
    def from_args(cls, args, modification=None):
        """Initialize this class from CLI arguments.

        Args:
            args: CLI arguments (as returned by argparse.ArgumentParser).
            modification: Dict keychain:value, specifying position values to be replaced in config dict.
        """
        assert hasattr(args, "config") and args.config is not None
        cfg_fname = Path(args.config)
        if hasattr(args, "resume") and args.resume is not None:
            resume = Path(args.resume)
        else:
            resume = None

        config = read_json(cfg_fname)

        return cls(config, modification=modification, resume=resume, use_mlflow=args.use_mlflow)

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self) -> Optional[Path]:
        return self._save_dir

    def __getitem__(self, name: str):
        """Access items like ordinary dict."""
        return self.config[name]

    def __contains__(self, item):
        return item in self.config

    # Methods to initialize tasks, model, data loaders, and trainer
    def init_tasks(self) -> OrderedDictType[str, ModalTask]:
        tasks = OrderedDict()
        tasks_args = self["tasks"]
        task2dataset2domains_ignored_in_metrics = None
        if "domains_ignored_in_metrics" in self:
            task2dataset2domains_ignored_in_metrics = \
                {task: {dataset: set(domains_ignored_in_metrics)
                        for dataset, domains_ignored_in_metrics in dataset2domains_ignored_in_metrics.items()}
                 for task, dataset2domains_ignored_in_metrics in self["domains_ignored_in_metrics"].items()}
        for task_name, file_path in tasks_args.items():
            file_path = prepend_optional_path(file_path, self.repo_path)
            task_args = read_json(file_path)
            dataset2domains_ignored_in_metrics = None
            if task2dataset2domains_ignored_in_metrics is not None and task_name in task2dataset2domains_ignored_in_metrics:
                dataset2domains_ignored_in_metrics = task2dataset2domains_ignored_in_metrics[task_name]
            tasks[task_name] = ModalTask.from_args_dict(task_name, task_args,
                                                        dataset2domains_ignored_in_metrics=dataset2domains_ignored_in_metrics,
                                                        repo_path=self.repo_path)

        return tasks

    def init_model(self, tasks: OrderedDictType[str, ModalTask]):

        """Initialize the model as specified in the configuration file."""
        model_type = self["model"]["type"]
        model_args = self["model"]["args"]

        return getattr(modules, model_type).from_args_dict(model_args, tasks, repo_path=self.repo_path)

    def init_data_loaders(self, model: modules.ModalClassifier) -> Dict[str, DataLoader]:
        """
        Initialize the data loaders as specified in the configuration file, and in such a way that they provide
        valid input for the given model.
        """
        params = self["data_loader"]
        data_loader_type = params["type"]
        # Make a copy to avoid storing old paths in new params
        data_loader_args = params["args"].copy()

        data_loaders = dict()
        for p in params["paths"]:
            data_loader_args["corpus_paths"] = params["paths"][p]
            if p == "val" or p == "test":
                # Do not shuffle data in validation and test data loader
                data_loader_args["shuffle"] = False
            data_loaders[p] = getattr(data_loaders_module, data_loader_type)(**data_loader_args, model=model)

        return data_loaders

    def init_data_sets(self, tasks: Dict[str, ModalTask]) -> Dict[str, SentenceDataset]:
        """
        Initializes data sets as specified in the configuration file. Only used for SVMBasedModalClassifier.
        """
        params = self["data_sets"]
        datasets = {p: SentenceDataset.from_corpus_files(paths, tasks) for p, paths in params["paths"].items()}

        return datasets

    def init_trainer(self, model: nn.Module, train_data_loader: DataLoader,
                     validation_data_loader: Optional[DataLoader] = None) -> Trainer:
        """Initialize the trainer for the given model. The model is trained on the specified train_data_loader and
        validated on the specified dev_data_loader.

        Args:
            model: Model to load data for.
            train_data_loader: Data loader for training data.
            validation_data_loader: Data loader for validation data.

        Returns:
            A trainer that trains the given model on data provided by the specified data loaders.
        """
        params = self["trainer"]

        optimizer = getattr(optim, params["optimizer"]["type"])(model.parameters(), **params["optimizer"]["args"])

        if "lr_scheduler" in params:
            lr_scheduler = self._init_lr_scheduler(optimizer, params["lr_scheduler"], train_data_loader)
        else:
            lr_scheduler = None

        return Trainer(model, optimizer, self.config, self.logger, self.save_dir, train_data_loader,
                       validation_data_loader, lr_scheduler=lr_scheduler, resume=self.resume, store_best=self.store_best)

    def _init_lr_scheduler(self, optimizer, params, train_data_loader: DataLoader):
        # Need train_data_loader as arg to eval "len(train_data_loader)" in scheduler_args["lr_lambda"]
        scheduler_type = params["type"]
        scheduler_args = params["args"]

        if "lr_lambda" in scheduler_args:
            if isinstance(scheduler_args["lr_lambda"], str):
                scheduler_args["lr_lambda"] = eval(scheduler_args["lr_lambda"])
            elif isinstance(scheduler_args["lr_lambda"], list):
                scheduler_args["lr_lambda"] = [eval(l) for l in scheduler_args["lr_lambda"]]
            else:
                raise Exception("lr_lambda must be a string or a list of strings")

        return getattr(lr_schedules_module, scheduler_type)(optimizer, **scheduler_args)


# Helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')