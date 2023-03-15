# This source code is from the PyTorch Template Project (w/ very heavy adaptations by Stefan Grünewald and Sophie Henning)
#   (https://github.com/victoresque/pytorch-template/blob/master/trainer/trainer.py)
# Copyright (c) 2018 Victor Huang
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1, MetricCollection

import numpy as np
import time
import os
from pathlib import Path
from math import inf
from typing import Optional, Dict, Tuple, List, Set

from logger.logger import Logger
from util import util
from data_handling.tasks import NO_DOMAINS
from data_handling.vocab import BasicVocab
from util.write_predictions import write_header, write_predictions_to_output_files


class Trainer:
    """
    Class for running the training.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, config: Dict, logger: Logger, save_dir: Path,
                 train_data_loader: DataLoader, validation_data_loader: Optional[DataLoader] = None, lr_scheduler=None,
                 resume: Optional[str] = None, store_best: bool = True):
        """

        Args:
            model:
            optimizer:
            config:
            train_data_loader:
            validation_data_loader:
        """
        self.config = config
        self.logger = logger

        # Set up GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            raise NotImplementedError(
                "Multi-GPU training is not implemented yet!")

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_data_loader = train_data_loader
        self.val_data_loader = validation_data_loader

        # Metrics for each modal, putting instances from all domains (if existing) together
        self.task2modal2dataset2metric_collection = dict()

        # Per-domain metrics
        self.task2domain2modal2dataset2metric_collection = dict()
        # Use torchmetrics to compute accuracy and weighted F1 scores,
        # manually compute per-class F1 scores based on tp, fp, and fn counts in weighted F1 scores

        tasks = self.model.tasks
        datasets = ["train", "val", "test"]
        for task_name, task in tasks.items():
            num_classes = len(task.output_vocab)
            metrics = MetricCollection([Accuracy(num_classes=num_classes).to(self.device),
                                        F1(num_classes=num_classes, average="weighted").to(self.device)])

            domains = task.target_domains
            modals = task.target_modals
            modal2dataset2metric_collection = dict()
            for modal in modals:
                dataset2metric_collection = dict()
                for dataset in datasets:
                    # Do not add domain name if there are no target domains
                    prefix = f"{task_name}_{dataset}_{modal}_"
                    dataset2metric_collection[dataset] = metrics.clone(
                        prefix=prefix)
                modal2dataset2metric_collection[modal] = dataset2metric_collection

            self.task2modal2dataset2metric_collection[task_name] = modal2dataset2metric_collection

            # Compile per-domain metrics if necessary - could also be done in the for-loops above, but doing it here
            # makes the code more readable
            if domains != NO_DOMAINS:
                domain2modal2dataset2metric_collection = dict()
                for domain in domains:
                    modal2dataset2metric_collection = dict()
                    for modal in modals:
                        dataset2metric_collection = dict()
                        for dataset in datasets:
                            if domain in task.dataset2domains_ignored_in_metrics[dataset]:
                                continue
                            prefix = f"{task_name}_{dataset}_{domain}_{modal}_"
                            dataset2metric_collection[dataset] = metrics.clone(
                                prefix=prefix)
                        modal2dataset2metric_collection[modal] = dataset2metric_collection
                    domain2modal2dataset2metric_collection[domain] = modal2dataset2metric_collection

                self.task2domain2modal2dataset2metric_collection[
                    task_name] = domain2modal2dataset2metric_collection

        self.start_epoch = 1
        # To compare with M&F's 1001 iterations (=steps)
        self.num_steps = 0

        trainer_config = config["trainer"]
        self.min_epochs = trainer_config["min_epochs"]
        self.max_epochs = trainer_config["max_epochs"]
        self.save_period = trainer_config["save_period"]
        self.accumulate_gradients = trainer_config["accumulate_gradients"]
        self.early_stopping = trainer_config.get("early_stopping", inf)

        # Gather info on validation metric (for early stopping and hyperparameter tuning)
        validation_metric_info = trainer_config["validation_metric"]
        validation_metric = validation_metric_info["type"]
        # Which tasks to consider in the validation metric
        self.tasks_in_validation_metric = tasks.keys()

        if "args" in validation_metric_info.keys():
            validation_metric_args = validation_metric_info["args"]
            if "tasks" in validation_metric_args.keys():
                self.tasks_in_validation_metric = set(
                    validation_metric_args["tasks"])
                if len(self.tasks_in_validation_metric) == 0:
                    raise ValueError(
                        f"Illegal Trainer configuration: at least one task has to be used in validation metric.")

        assert validation_metric in {"accuracy", "weighted_F1"}
        self.validation_metric = validation_metric
        self.validation_metric_general_name = util.get_general_name_of_validation_metric(self.tasks_in_validation_metric,
                                                                                         validation_metric)
        self.validation_metric_final_name = f"val_{self.validation_metric_general_name}"

        # Compute number of combinations of task, domains, and modal verbs to get the correct denominator
        # for macro-averaging
        # If multi-task: only consider those tasks that are supposed to be considered
        self.dataset2denominator = {dataset: 0 for dataset in datasets}
        for task in tasks.values():
            if task.name in self.tasks_in_validation_metric:
                for dataset in datasets:
                    num_domain_modal_combs_ignored = \
                        len(task.dataset2domains_ignored_in_metrics[dataset]) * len(
                            task.target_modals)
                    self.dataset2denominator[dataset] += task.num_domain_modal_combs - \
                        num_domain_modal_combs_ignored

        self.checkpoint_dir = save_dir
        self.store_best = store_best
        if resume is not None:
            self.resume_checkpoint(resume)

    def run_epoch(self, epoch: int, data_loader: DataLoader, use_deterministic_algorithms: bool,
                  training: bool = False) -> Dict[str, float]:
        """
        Run one epoch.

        Args:
            epoch: Number of current epoch.
            data_loader: Data loader to fetch examples from.
            use_deterministic_algorithms: use deterministic PyTorch algorithms
            training: If true, model will be trained. Otherwise, the model will simply be evaluated. Default: False.

        Returns:
        """
        # Set model to train or eval mode
        if training:
            self.model.train()
        else:
            self.model.eval()

        # Set epoch metrics
        loss_string = "train_loss" if training else "val_loss"
        epoch_metrics = {loss_string: 0}

        # Track epoch progress
        num_completed_batches = 0

        with torch.set_grad_enabled(training):
            for step, batch in enumerate(data_loader):
                sentences, targets, task_names, modal_indices, domain_indices = batch

                if targets.device != self.device:
                    targets = self._to_device(targets)

                # Perform a forward pass
                mode = "training" if training else "validation"
                predictions, batch_loss = self.model(sentences, task_names, modal_indices, domain_indices, self.device,
                                                     mode, targets)
                epoch_metrics[loss_string] += batch_loss.item()

                # Retrieve per-task per-domain per-modal predictions and targets to be able to compute
                # per-task per-domain per-modal metrics
                task2domain2modal2preds, task2domain2modal2tgts, _, _ = \
                    self._get_per_task_per_domain_per_modal_tensors(predictions, targets, task_names,
                                                                    modal_indices, domain_indices)

                if training:
                    # sentences_str = '|'.join([' '.join(sentence.tokens) for sentence in sentences])
                    # self.logger.debug(f"Sentences in training epoch {epoch}, batch {num_completed_batches + 1}: "
                    #                   f"{sentences_str}")
                    self._update_per_task_per_domain_per_modal_metrics(task2domain2modal2preds, task2domain2modal2tgts,
                                                                       "train")

                    # Perform a backward pass to calculate gradients
                    torch.use_deterministic_algorithms(False)
                    batch_loss.backward()
                    torch.use_deterministic_algorithms(
                        use_deterministic_algorithms)

                    # Update parameters every k-th batch
                    current_batch_num = num_completed_batches + 1
                    if current_batch_num % self.accumulate_gradients == 0:
                        self.optimizer.step()
                        # Zero gradients for next batch
                        self.optimizer.zero_grad()
                        self.logger.debug(
                            f"Updated parameters after {current_batch_num}-th batch")

                        # Only increase number of steps when training
                        self.num_steps += 1
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                            lrs = ", ".join(
                                f"{lr:.2e}" for lr in self.lr_scheduler.get_lr())
                            self.logger.info(f"LRs are now: {lrs}")

                else:
                    self._update_per_task_per_domain_per_modal_metrics(task2domain2modal2preds, task2domain2modal2tgts,
                                                                       "val")

                num_completed_batches += 1
                self.logger.debug(f"{'Training' if training else 'Validation'} Epoch: {epoch} "
                                  f"{util.pretty_print_ratio(num_completed_batches, len(data_loader))} of batches "
                                  f"completed Loss: {batch_loss:.6f}")

        # Compute average loss and other epoch metrics
        epoch_metrics[loss_string] = np.mean(epoch_metrics[loss_string])
        dataset = "train" if training else "val"

        # Compute per-class F1 first (to be able to reset metrics in _compute_per_modal_metrics)
        epoch_metrics.update(
            self._compute_per_label_f1_scores_for_all_tasks(dataset))

        # Compute overall per-domain per-modal metrics
        epoch_metrics.update(
            self._compute_per_domain_per_modal_metrics_and_macro(dataset))

        return epoch_metrics

    def train(self, use_deterministic_algorithms: bool) -> Tuple[float, float, Dict]:
        """Train the model"""
        training_starttime = time.time()

        # Track metric of best model on validation set
        val_metric_best = 0
        val_loss_best = inf
        val_other_metrics_of_best_model = dict()
        # For early stopping after k epochs: count how many epochs not improved on val set
        not_improved_count = 0
        stopped_early = False

        # Sanity check: monitor loss on validation set before any training
        if self.val_data_loader:
            epoch_starttime = time.time()
            val_epoch_metrics = self.run_epoch(
                0, self.val_data_loader, use_deterministic_algorithms)
            epoch_duration = (time.time() - epoch_starttime) / 60
            self.logger.info(
                f"Validation epoch {0} finished. Duration: {epoch_duration:.1f} minutes.")
            self.logger.log_epoch_metrics(val_epoch_metrics, 0)

        for epoch_i in range(self.start_epoch, self.max_epochs+1):
            # Training
            epoch_starttime = time.time()
            train_epoch_metrics = self.run_epoch(epoch_i, self.train_data_loader, use_deterministic_algorithms,
                                                 training=True)
            epoch_duration = (time.time() - epoch_starttime) / 60
            self.logger.info(
                f"Training epoch {epoch_i} finished. Duration: {epoch_duration:.1f} minutes.")
            self.logger.log_epoch_metrics(train_epoch_metrics, epoch_i)
            self.logger.info(f"Total number of steps so far: {self.num_steps}")

            # Validation
            if self.val_data_loader:
                epoch_starttime = time.time()
                val_epoch_metrics = self.run_epoch(
                    epoch_i, self.val_data_loader, use_deterministic_algorithms)
                epoch_duration = (time.time() - epoch_starttime) / 60
                self.logger.info(
                    f"Validation epoch {epoch_i} finished. Duration: {epoch_duration:.1f} minutes.")
                self.logger.log_epoch_metrics(val_epoch_metrics, epoch_i)

                current_val_metric = val_epoch_metrics[self.validation_metric_final_name]

                if current_val_metric > val_metric_best:
                    # New best model
                    val_metric_best = current_val_metric
                    val_loss_best = val_epoch_metrics["val_loss"]
                    val_other_metrics_of_best_model = {key: value for key, value in val_epoch_metrics.items()
                                                       if key != "val_loss" and key != self.validation_metric_final_name}
                    if self.store_best:
                        self._save_checkpoint(epoch_i, is_best=True)

                    # The model just improved
                    not_improved_count = 0
                else:
                    # No model improvement in this epoch
                    if epoch_i % self.save_period == 0:
                        # Regular checkpoint
                        self._save_checkpoint(epoch_i, is_best=False)

                    not_improved_count += 1
                if not_improved_count > self.early_stopping and epoch_i >= self.min_epochs:
                    stopped_early = True
                    break

            else:
                # No validation - just store the last model as best for later evaluation
                if epoch_i == self.max_epochs:
                    self._save_checkpoint(epoch_i, is_best=True)

        if self.val_data_loader:
            if stopped_early:
                self.logger.info(
                    f"{self.validation_metric_final_name} did not improve for {self.early_stopping} epochs. Training stops.")
            else:
                self.logger.info(
                    "Maximum epoch number reached. Training stops.")
        else:
            self.logger.info("Maximum epoch number reached. Training stops.")

        training_duration = (time.time() - training_starttime) / 60
        self.logger.info(f"Training took {training_duration:.1f} minutes.")

        if self.val_data_loader:
            self.logger.log_metric(
                f"best_{self.validation_metric_final_name}", val_metric_best)
            self.logger.log_metric(
                "val_loss_of_best_model", val_loss_best, percent=False)

        return val_metric_best, val_loss_best, val_other_metrics_of_best_model

    def test(self, eval_data_loader: DataLoader, output_paths: Dict[str, Path]) -> Dict:
        """Run model on evaluation set, storing per-task predictions in a single file each."""
        util.set_seed(42)
        test_starttime = time.time()
        tasks = self.model.tasks

        # Write header in each task prediction file
        write_header(eval_data_loader.corpus_paths, tasks, output_paths)

        # Make predictions on test set
        with torch.no_grad():
            self.model.eval()
            for step, batch in enumerate(eval_data_loader):
                sentences, targets, task_names, modal_indices, domain_indices = batch

                if targets.device != self.device:
                    targets = targets.to(self.device)

                # Perform a forward pass
                predictions, confidences = self.model(
                    sentences, task_names, modal_indices, domain_indices, self.device)

                # Map task-domain-modal combination to predictions/targets/confidences
                task2domain2modal2preds, task2domain2modal2tgts, task2domain2modal2mask, task2domain2modal2confs = \
                    self._get_per_task_per_domain_per_modal_tensors(predictions, targets, task_names,
                                                                    modal_indices, domain_indices, confidences)

                # Update metrics
                self._update_per_task_per_domain_per_modal_metrics(task2domain2modal2preds, task2domain2modal2tgts,
                                                                   "test")
                # Write predictions to output file
                write_predictions_to_output_files(sentences, task2domain2modal2preds, task2domain2modal2tgts,
                                                  task2domain2modal2confs, task2domain2modal2mask, tasks, output_paths)

        # Log the predictions as MLFlow artifacts
        for output_path in output_paths.values():
            self.logger.log_artifact(output_path)

        # Compute per-label F1 scores
        test_metrics = self._compute_per_label_f1_scores_for_all_tasks("test")

        for task_name, task in tasks.items():
            if not task.single_label_task:
                continue

            # Also return number of correct and number of total predictions for compatibility with Marasovic/Fr MASC
            # evaluation
            # We cannot simply return the torchmetrics Accuracy and F1 objects, as the main function of train runs in a
            # subprocess when doing CV evaluation and trying to pass the Accuracy and F1 objects from the subprocess to
            # the main process leads to a 'CUDA error: invalid resource handle' when trying to rebuild the associated
            # tensors

            # Accuracy object counts true positives and false negatives across classes
            domains = task.target_domains
            modals = task.target_modals

            for modal in modals:
                modal_metrics = self.task2modal2dataset2metric_collection[task_name][modal]["test"]
                modal_accuracy = modal_metrics["Accuracy"]
                num_correct_string = f"{task_name}_{modal}_num_correct"
                test_metrics[num_correct_string] = modal_accuracy.tp.item()
                test_metrics[f"{task_name}_{modal}_num_total"] = test_metrics[num_correct_string] \
                    + modal_accuracy.fn.item()
                if domains != NO_DOMAINS:
                    for domain in domains:
                        if domain in task.dataset2domains_ignored_in_metrics["test"]:
                            continue

                        domain_metrics = \
                            self.task2domain2modal2dataset2metric_collection[
                                task_name][domain][modal]["test"]
                        domain_accuracy = domain_metrics["Accuracy"]
                        num_correct_string = f"{task_name}_{domain}_{modal}_num_correct"
                        test_metrics[num_correct_string] = domain_accuracy.tp.item()
                        test_metrics[f"{task_name}_{modal}_num_total"] = test_metrics[num_correct_string] \
                            + domain_accuracy.fn.item()

        test_metrics.update(
            self._compute_per_domain_per_modal_metrics_and_macro("test")
        )

        self.logger.log_epoch_metrics(test_metrics)

        test_duration = (time.time() - test_starttime) / 60
        self.logger.info(f"Testing took {test_duration:.1f} minutes.")

        return test_metrics

    def _get_per_task_per_domain_per_modal_tensors(self, predictions: torch.Tensor,
                                                   targets: torch.Tensor,
                                                   tasks_in_batch: List[str],
                                                   modal_indices: List[int],
                                                   domain_indices: List[int],
                                                   confidences: Optional[torch.Tensor] = None) \
            -> Tuple[Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]],
                     Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]],
                     Dict[str, Dict[str, Dict[str, List[int]]]],
                     Optional[Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]]]]:
        """
        Map task-domain-modal combinations to tensors containing only the corresponding predictions and targets (and confidences, if this argument is passed).
        Expects predictions and targets tensor to be of the same shape (batch_size, num_labels+padding (self.model.per_head_padded_preds_width)), 
        i.e., the predictions tensor already contains only the predictions of the respectively correct head.
        Also return a dictionary mapping each task-domain-modal combination to a list of batch indices 
        of instances of the given combination ("mask").
        """
        task2domain2modal2predictions, task2domain2modal2targets, task2domain2modal2mask = {}, {}, {}
        task2domain2modal2confidences = None if confidences is None else {}
        tasks = self.model.tasks

        for task_name, task in tasks.items():
            # Step 1: Gather task-specific information, set up dicts
            modals = task.target_modals
            domains = task.target_domains
            single_label_task = task.single_label_task
            task2domain2modal2predictions[task_name] = dict()
            task2domain2modal2targets[task_name] = dict()

            # Step 2: Select the per-task columns from the cross-task predictions/targets/confidences
            # Compute the column indices for predictions and targets (they have the same shape)
            start, end = self.model.tasks2per_head_preds_span[task_name]

            task_tgts = targets[:, start:end]
            task_preds = predictions[:, start:end]
            if single_label_task:
                task_tgts = task_tgts.flatten()
                task_preds = task_preds.flatten().long()

            # Repeat steps 1-2 for confidences if applicable
            if task2domain2modal2confidences is not None:
                task2domain2modal2confidences[task_name] = dict()
                start_confs, end_confs = self.model.tasks2per_head_confs_span[task_name]
                task_confs = confidences[:, start_confs:end_confs]

            # Step 3: Compute masks for selecting per-domain per-modal predictions/targets (= list of batch indices)
            domain2modal2mask = {domain: {modal: [] for modal in modals}
                                 for domain in domains}

            for j in range(len(modal_indices)):
                # Restrict mask to task predictions/targets/confidences
                if tasks_in_batch[j] != task_name:
                    continue

                # Retrieve the target modal
                modal = modals[modal_indices[j]]

                # Retrieve the target domain
                domain = domains[domain_indices[j]]

                # j is the instance's index in the batch
                domain2modal2mask[domain][modal].append(j)

            # Step 4: Retrieve the actual predictions/targets/confidences using the pre-compiled masks
            for domain in domains:
                task2domain2modal2predictions[task_name][domain] = dict()
                task2domain2modal2targets[task_name][domain] = dict()
                if task2domain2modal2confidences is not None:
                    task2domain2modal2confidences[task_name][domain] = dict()

                for modal in modals:
                    mask = domain2modal2mask[domain][modal]
                    if len(mask) == 0:
                        # No instance of this domain-modal combination in the given batch
                        task2domain2modal2predictions[task_name][domain][modal] = None
                        task2domain2modal2targets[task_name][domain][modal] = None
                    else:
                        task2domain2modal2predictions[task_name][domain][modal] = task_preds[mask]
                        task2domain2modal2targets[task_name][domain][modal] = task_tgts[mask]

                    if task2domain2modal2confidences is not None:
                        task2domain2modal2confidences[task_name][domain][modal] = None if len(
                            mask) == 0 else task_confs[mask]

            task2domain2modal2mask[task_name] = domain2modal2mask

        return task2domain2modal2predictions, task2domain2modal2targets, task2domain2modal2mask, task2domain2modal2confidences

    def _update_per_task_per_domain_per_modal_metrics(
            self, task2domain2modal2preds: Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]],
            task2domain2modal2tgts: Dict[str, Dict[str, Dict[str, Optional[torch.Tensor]]]], dataset: str)\
            -> None:

        for task_name, task in self.model.tasks.items():
            for domain in task.target_domains:
                for modal in task.target_modals:
                    modal_preds = task2domain2modal2preds[task_name][domain][modal]
                    if modal_preds is not None:
                        modal_targets = task2domain2modal2tgts[task_name][domain][modal]
                        modal_metrics = self.task2modal2dataset2metric_collection[
                            task_name][modal][dataset]
                        modal_metrics(modal_preds, modal_targets)

                        # Update domain metrics if domains exist
                        if task.target_domains != NO_DOMAINS:
                            if domain in task.dataset2domains_ignored_in_metrics[dataset]:
                                continue

                            domain_metrics = \
                                self.task2domain2modal2dataset2metric_collection[
                                    task_name][domain][modal][dataset]
                            domain_metrics(modal_preds, modal_targets)

    def _compute_metrics_in_metric_collection(self, metric_collection: MetricCollection, result: Dict[str, float],
                                              sum_weighted_f1_scores: float, task_name: str, dataset: str, modal: str,
                                              add_values_to_sum_weighted_f1_scores: bool = True,
                                              reset_metrics: bool = True, domain: Optional[str] = None) \
            -> Tuple[float, Dict[str, float]]:
        """Helper function for computing all metrics in a metric collection"""
        task_dataset = f"{task_name}_{dataset}"

        for metric_name, metric in metric_collection.items():
            value = metric.compute().item()
            metric_with_average = f"{metric.average}_{metric_name}"
            # (f"Computed metric {metric_name}, value: {value}")
            prefix = f"{task_dataset}_{metric_with_average}"
            if domain is not None:
                prefix = f"{prefix}_{domain}"
            result[f"{prefix}_{modal}"] = value
            if metric_with_average == "weighted_F1":
                if task_name in self.tasks_in_validation_metric and add_values_to_sum_weighted_f1_scores:
                    sum_weighted_f1_scores += value
                    # self.logger.debug(f"Sum of weighted F1 scores now is {sum_weighted_f1_scores}")

            if reset_metrics:
                metric.reset()
                # self.logger.debug("Reset metric.")

        return sum_weighted_f1_scores, result

    def _compute_per_domain_per_modal_metrics_and_macro(self, dataset: str, reset_metrics: bool = True)\
            -> Dict[str, float]:
        """
        Compute per-domain per-modal metrics collected for dataset and macro-average across all domain-modal combinations.
        """

        result = dict()
        tasks = self.model.tasks

        # For model selection: compute macro average over per-task per-domain per-modal weighted F1 scores
        sum_weighted_f1_scores = 0

        for task_name, task in tasks.items():
            domains = task.target_domains
            has_domains = domains != NO_DOMAINS
            modals = task.target_modals

            for modal in modals:
                modal_metric_collection = self.task2modal2dataset2metric_collection[
                    task_name][modal][dataset]
                sum_weighted_f1_scores, result = \
                    self._compute_metrics_in_metric_collection(modal_metric_collection, result, sum_weighted_f1_scores,
                                                               task_name, dataset, modal, not has_domains,
                                                               reset_metrics)
                if has_domains:
                    for domain in domains:
                        if domain in task.dataset2domains_ignored_in_metrics[dataset]:
                            continue
                        domain_metric_collection = \
                            self.task2domain2modal2dataset2metric_collection[
                                task_name][domain][modal][dataset]
                        sum_weighted_f1_scores, result = \
                            self._compute_metrics_in_metric_collection(domain_metric_collection, result,
                                                                       sum_weighted_f1_scores, task_name, dataset,
                                                                       modal, has_domains, reset_metrics, domain)

        # Compute an average of weighted F1 scores across tasks, possibly domains, and modals (for early stopping)
        avg_name = f"{dataset}_{self.validation_metric_general_name}"

        weighted_f1_avg = sum_weighted_f1_scores / \
            self.dataset2denominator[dataset]

        result[avg_name] = weighted_f1_avg

        return result

    def _compute_per_label_f1(self, f1_metric: nn.Module, output_vocab: BasicVocab, illegal_combinations: Set[str],
                              labels_ignored_in_modeling: Optional[Set[str]], metric_prefix: str, modal: str) \
            -> Dict[str, float]:
        """Compute per-label F1 scores for a given modal."""

        per_label_f1_scores = dict()
        # Compute per-label F1 scores based on counts in weighted F1 metric
        # F1 = tp/(tp+0.5*(fp+fn))
        per_label_denominator = f1_metric.tp + \
            0.5 * (f1_metric.fp + f1_metric.fn)
        # Replace 0 denominators with 1 (numerator will be 0 anyways)
        per_label_denominator = torch.where(per_label_denominator != 0, per_label_denominator,
                                            torch.ones(per_label_denominator.shape).to(self.device))

        # Rank-1 tensor of shape output_vocab_size
        per_label_metrics = f1_metric.tp / per_label_denominator

        for i in range(len(output_vocab)):
            label = output_vocab.ix2token(i)

            # Check if label is illegal for modal
            if label in illegal_combinations:
                continue

            # Ignore label if ignored in src
            if labels_ignored_in_modeling is not None:
                if label in labels_ignored_in_modeling:
                    continue

            # Everything fine
            per_label_f1_scores[f"{metric_prefix}_F1_{modal}_{label}"] = per_label_metrics[i].item(
            )

        return per_label_f1_scores

    def _compute_per_label_f1_scores_for_all_tasks(self, dataset: str) -> Dict[str, float]:
        f1_scores = dict()
        tasks = self.model.tasks

        for task_name, task in tasks.items():
            modals = task.target_modals
            domains = task.target_domains
            illegal_combinations = task.illegal_combinations
            modal_label_combinations_ignored_in_modeling = task.modal_label_combinations_ignored_in_modeling
            output_vocab = task.output_vocab
            task_dataset = f"{task_name}_{dataset}"

            for modal in modals:
                f1_metric = self.task2modal2dataset2metric_collection[task_name][modal][dataset]["F1"]
                modal_label_combinations_ignored_in_modeling_for_modal = \
                    None if modal_label_combinations_ignored_in_modeling is None else modal_label_combinations_ignored_in_modeling[modal]
                f1_scores.update(
                    self._compute_per_label_f1(f1_metric, output_vocab, illegal_combinations[modal],
                                               modal_label_combinations_ignored_in_modeling_for_modal, task_dataset, modal)
                )

                # Compute per-domain scores
                if domains != NO_DOMAINS:
                    for domain in domains:
                        if domain in task.dataset2domains_ignored_in_metrics[dataset]:
                            continue

                        metric_prefix = f"{task_dataset}_{domain}"
                        f1_metric = self.task2domain2modal2dataset2metric_collection[
                            task_name][domain][modal][dataset]["F1"]
                        f1_scores.update(
                            self._compute_per_label_f1(f1_metric, output_vocab, illegal_combinations[modal],
                                                       modal_label_combinations_ignored_in_modeling_for_modal, metric_prefix, modal))

        return f1_scores

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save a checkpoint.

        Args:
            epoch: number of current epoch.
            is_best: If True, rename the saved checkpoint to 'model_best.pth'.
        """
        architecture = type(self.model).__name__

        state = {
            'arch': architecture,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir,
                                f"checkpoint-epoch{epoch}.pth")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
            self.logger.info(f"Saving current best checkpoint: {best_path}...")
            torch.save(state, best_path)

        else:
            self.logger.info(f"Saving regular checkpoint: {filename}...")
            torch.save(state, filename)

    def resume_checkpoint(self, resume_path: str):
        """
        Resume from saved checkpoint.

        Args:
            resume_path: checkpoint path to be resumed
        """
        self.logger.info(f"Loading checkpoint: {resume_path}...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['trainer']['optimizer']['type'] != self.config['trainer']['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(
            f"Checkpoint loaded. Resume from epoch {self.start_epoch}")

    def _prepare_device(self, n_gpu_use: int) -> Tuple[torch.device, List[int]]:
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _to_device(self, data):
        if isinstance(data, torch.Tensor):
            assert data.device != self.device
            return data.to(self.device)
        elif isinstance(data, dict):
            assert all(isinstance(val, torch.Tensor) for val in data.values())
            assert all(val.device != self.device for val in data.values())
            data_on_device = dict()
            for key in data:
                data_on_device[key] = data[key].to(self.device)
            return data_on_device
        else:
            raise Exception("Cannot move this kind of data to a device!")
