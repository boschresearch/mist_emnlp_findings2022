# This source code is from the STEPS Parser (w/ heavy adaptations by Sophie Henning)
#   (https://github.com/boschresearch/steps-parser/blob/master/src/train.py)
# Copyright (c) 2020 Robert Bosch GmbH
# This source code is licensed under the AGPL v3 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
# Author: Stefan Grünewald

import argparse
import json
import os
import torch

from pathlib import Path
from multiprocessing import Queue

from config.config_parser import ConfigParser

from typing import Dict, List, Optional

from util.util import set_seed
from data_handling.tasks import ModalTask


class Result:
    def __init__(self, best_val_metric: float, val_loss_of_best_model: float, other_metrics_of_best_model: Dict,
                 val_metric: str, tasks: Dict[str, ModalTask]):
        self.best_val_metric = best_val_metric
        self.val_loss_of_best_model = val_loss_of_best_model
        self.other_metrics_of_best_model = other_metrics_of_best_model
        self.val_metric = val_metric
        self.test_results: Optional[Dict] = None
        self.tasks = tasks


def main(config_parser: ConfigParser, evaluate: bool = False, result_queue: Optional[Queue] = None)\
        -> Result:
    """Main function to initialize model, load data, and run training.

    Args:
        config_parser: Experiment configuration.
        evaluate: whether to run the trained model on the test set with more detailed statistics
        result_queue: Result object will be put into result_queue if this argument is specified

    Returns:
        the training and possibly evaluation results, stored in a Result object
    """
    # Use deterministic algorithms if wanted
    use_deterministic_algorithms = False
    if config_parser.config["use_deterministic_algorithms"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        use_deterministic_algorithms = True
        torch.use_deterministic_algorithms(True)

    # Set seed for reproducibility
    set_seed(config_parser.config["random_seed"])

    tasks = config_parser.init_tasks()

    model = config_parser.init_model(tasks)

    data_loaders = config_parser.init_data_loaders(model)

    if "val" in data_loaders.keys():
        trainer = config_parser.init_trainer(model, data_loaders["train"], data_loaders["val"])
    else:
        trainer = config_parser.init_trainer(model, data_loaders["train"])

    best_val_metric, val_loss_of_best_model, other_metrics_of_best_model = trainer.train(use_deterministic_algorithms)

    result = Result(best_val_metric, val_loss_of_best_model, other_metrics_of_best_model,
                    trainer.validation_metric_final_name, trainer.model.tasks)

    if evaluate:
        eval_dataloader = data_loaders["test"]

        # Load best checkpoint for model
        checkpoint_dir_path = Path(trainer.checkpoint_dir)
        checkpoint_path = checkpoint_dir_path / "model_best.pth"
        trainer.resume_checkpoint(checkpoint_path)

        # Generate output file paths for storing the predictions on the test set for each task
        output_files: Dict[str, Path] = {}
        for task_name in trainer.model.tasks.keys():
            output_files[task_name] = checkpoint_dir_path / f"{task_name}_eval.txt"

        result.test_results = trainer.test(eval_dataloader, output_files)

    if result_queue is not None:
        result_queue.put(result)

    return result


def init_config_modification(raw_modifications: List[str]) -> Dict:
    """Turn a "raw" config modification string into a dictionary of key-value pairs to replace."""
    modification = dict()
    for mod in raw_modifications:
        key, value = mod.split("=", 1)
        modification[key] = json.loads(value)

    return modification


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='ModalFunctionsClassifier (training mode)')
    argparser.add_argument('config', type=str, help='config file path (required)')
    argparser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    argparser.add_argument('-s', '--save-dir', default=None, type=str, help='model save directory (config override)')
    argparser.add_argument('-m', '--modification', default=None, type=str, nargs='+',
                           help='modifications to make to specified configuration file (config override)')
    argparser.add_argument('-u', '--use-mlflow', action='store_true', help='Use MLFlow')
    argparser.add_argument('-e', '--evaluate', action='store_true', help='Evaluate on test corpus')

    # Parse arguments and apply possible modifications to config
    args = argparser.parse_args()
    modification = init_config_modification(args.modification) if args.modification is not None else dict()
    if args.save_dir is not None:
        modification["trainer.save_dir"] = args.save_dir

    config = ConfigParser.from_args(args, modification=modification)
    main(config, evaluate=args.evaluate)