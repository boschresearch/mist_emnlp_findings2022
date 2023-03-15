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

from collections import defaultdict
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
from .vocab import BasicVocab
from config.config_utils import read_json
from util.util import prepend_optional_path

NO_DOMAINS = ["dummy"]


class ModalTask:
    """
    A modal classification task.
    Attributes:
        name: Name of the task.
        single_label_task: Is this task a single-label task?
        has_documents: does the corpus consist of different documents?
        target_modals:  List of modals to be classified in the task
        illegal_combinations: Dict from modal to set of illegal labels for this modal
        modal_label_combinations_ignored_in_modeling: Dict from modal to set of labels which are ignored
        target_domains: List of domains in the task, set to constant NO_DOMAINS if there are no domains in the given task
    """
    def __init__(self, name: str, single_label_task: bool, has_documents: bool, output_vocab: BasicVocab,
                 target_modals: List[str], illegal_combinations: Dict[str, Set[str]],
                 modal_label_combinations_ignored_in_modeling: Optional[Dict[str, Set[str]]],
                 target_domains: Optional[List[str]],
                 dataset2domains_ignored_in_metrics: Optional[Dict[str, Set[str]]]):
        """
        Args:
            name: Name of the task
            single_label_task: Is this task a single-label task?
            output_vocab: label vocabulary of the task
            target_modals:  List of modals to be classified in the task
            illegal_combinations: Dict from modal to set of illegal labels for this modal
            modal_label_combinations_ignored_in_modeling: Dict from modal to set of labels which are ignored
            target_domains: List of domains in the task, should be None if there are no domains in the given task
            dataset2domains_ignored_in_metrics: optional, use this if you do not want to compute metrics for certain
            domains on the train, val, or test dataset -- used for cross-domain experiments where the test data
            consists only of the data of a specific domain and the train/val data consists of the remaining domains
        """
        self.name = name
        self.single_label_task = single_label_task  # Is this a single-label task?
        self.has_documents = has_documents
        self.target_modals = target_modals  # Modal verbs in the task
        self.target_domains = NO_DOMAINS if target_domains is None else target_domains  # Domains of the task
        # Domains ignored on a specific dataset - use a defaultdict for a simpler look-up
        self.dataset2domains_ignored_in_metrics = defaultdict(set)
        if dataset2domains_ignored_in_metrics is not None:
            for dataset, domains_ignored_in_metrics in dataset2domains_ignored_in_metrics.items():
                self.dataset2domains_ignored_in_metrics[dataset] = domains_ignored_in_metrics

        self.output_vocab = output_vocab
        self.labels = output_vocab.get_real_tokens()

        # Illegal combinations: modal-label combinations illegal by the annotation scheme
        self.illegal_combinations = illegal_combinations  # Illegal labels per modal

        # Modal-label combinations that are ignored in src
        # Need to keep the distinction between illegal and non-occurring combinations, as there are 3 instances of
        # "may" labeled "dynamic" in MASC (which is an illegal combination), which we kept in the data to ensure
        # comparibility of our results in terms of accuracy (but we do not compute a per-label F1 for may-dy)
        # Modal -> Set(labels)
        self.modal_label_combinations_ignored_in_modeling = modal_label_combinations_ignored_in_modeling

        # Each domain has instances of each modal
        self.num_domain_modal_combs = len(self.target_domains) * len(self.target_modals)

    def __str__(self):
        return f"ModalTask({self.name})"

    @classmethod
    def from_args_dict(cls, task_name: str, task_args: Dict, replace_path: Optional[Tuple[str, str]] = None,
                       dataset2domains_ignored_in_metrics: Optional[Dict[str, Set[str]]] = None,
                       repo_path: Optional[Path] = None):
        ov_args = task_args["output_vocab"]
        output_vocab_type = ov_args["type"]
        if output_vocab_type == "BasicVocab":
            inner_args = ov_args["args"]
            file_path = inner_args["vocab_filename"]
            if replace_path is not None:
                file_path = file_path.replace(replace_path[0], replace_path[1])
            file_path = prepend_optional_path(file_path, repo_path)
            output_vocab = BasicVocab(file_path)
        else:
            raise NotImplementedError(f"Output vocab type {output_vocab_type} not implemented for task output "
                                      f"vocabs")

        single_labeling = task_args["single_labeling"]
        has_documents = task_args["has_documents"]
        target_modals = task_args["target_modals"]

        # Retrieve illegal modal-label combinations if there are any
        illegal_combinations = set()
        if "illegal_combinations" in task_args.keys():
            illegal_combinations_json = task_args["illegal_combinations"]
            illegal_combinations = {modal: set(illegal_combinations_json[modal])
                                    if modal in illegal_combinations_json.keys() else set()
                                    for modal in target_modals}

        # Read in ignored modal-label combinations if they are specified
        modal_label_combinations_ignored_in_modeling = None
        if "modal_label_combinations_ignored_in_modeling" in task_args.keys():
            # Non-occurring combinations are stored in an automatically generated JSON file,
            # the task config contains the path to this file
            file_path = task_args["modal_label_combinations_ignored_in_modeling"]
            if replace_path is not None:
                file_path = file_path.replace(replace_path[0], replace_path[1])

            file_path = prepend_optional_path(Path(file_path), repo_path)
            modal_label_combinations_ignored_in_modeling_json = read_json(file_path)
            modal_label_combinations_ignored_in_modeling = {modal: set(rare_labels)
                                          for modal, rare_labels in modal_label_combinations_ignored_in_modeling_json.items()}

        target_domains = None
        if "target_domains" in task_args.keys():
            target_domains = task_args["target_domains"]

        return ModalTask(task_name, single_labeling, has_documents, output_vocab, target_modals, illegal_combinations,
                         modal_label_combinations_ignored_in_modeling, target_domains,
                         dataset2domains_ignored_in_metrics=dataset2domains_ignored_in_metrics)