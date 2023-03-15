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

import torch
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from typing import OrderedDict as OrderedDictType #  Avoid confusion with collections.OrderedDict

import modules.embedders as embedders
import trainer.losses as losses
from data_handling.sentence_with_modal_sense import SentenceWithModalSense
from data_handling.vocab import BasicVocab, UNARY_VOCAB
from data_handling.tasks import ModalTask, NO_DOMAINS
from util.util import prepend_optional_path


class Head(nn.Module):
    def __init__(self, layer_type: str, dropout: float, input_dim: int, output_dim: int, use_sigmoid: bool,
                 sigmoid_threshold: Optional[float] = None) -> None:
        super(Head, self).__init__()
        self.layer = getattr(nn, layer_type)(input_dim, output_dim)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=dropout)

        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.activation = nn.Sigmoid()
            if sigmoid_threshold is not None:
                self.sigmoid_threshold = sigmoid_threshold
            else:
                self.sigmoid_threshold = 0.5
        else:
            self.activation = nn.Softmax(dim=1)

    @classmethod
    def from_args_dict(cls, args_dict: Dict, input_dim: int, output_dim: int):
        single_labeling = args_dict["single_labeling"]
        use_sigmoid = (not single_labeling) or output_dim == 1
        sigmoid_threshold = None
        if "sigmoid_threshold" in args_dict.keys():
            sigmoid_threshold = args_dict["sigmoid_threshold"]

        return cls(args_dict["type"], args_dict["args"]["dropout"], input_dim, output_dim, use_sigmoid,
                   sigmoid_threshold)

    def forward(self, embedded: torch.Tensor):
        return self.layer(self.dropout(embedded))


class ModalClassifier(nn.Module):
    def __init__(self, embedder: embedders.Embedder, heads: nn.ModuleDict, non_task_output_vocabs: Dict[str, BasicVocab],
                 tasks: OrderedDictType[str, ModalTask], single_label_loss_fn: str, multi_label_loss_fn: str,
                 use_per_modal_heads: bool = True, use_per_domain_heads: bool = False):
        super(ModalClassifier, self).__init__()
        self.embedder = embedder
        self.heads = heads
        self.non_task_output_vocabs = non_task_output_vocabs
        self.use_per_modal_heads = use_per_modal_heads
        self.use_per_domain_heads = use_per_domain_heads

        self.tasks = tasks
        # Map each task to its loss function
        self.tasks2losses = nn.ModuleDict()
        for name, task in self.tasks.items():
            self.tasks2losses[name] = getattr(losses, single_label_loss_fn)() if task.single_label_task \
                else getattr(losses, multi_label_loss_fn)()

        # ---------------------------Precomputing for multi-task architecture ------------------------------------------
        # Idea: Since the tasks can be mixed in each batch, the whole batch needs to be processed by all task-specific
        # heads. We then stack the task-specific output matrices in order to obtain an output tensor with shape
        # (x, batch_size, padded_output_length), where x is the number of combinations of tasks, domains and modals
        # for a model with per-domain per-modal heads, the number of combinations of tasks and modals for a model with
        # per-modal heads, and the number of tasks for a modal with a single head for each task.
        # For each row in the input matrix, we then want to select the output for the right task
        # (i.e., the one the input example comes from). To be able to select these instances in a vectorized fashion,
        # the task outputs all need to be of the same size (even if their output vocabulary sizes differ).

        # Step 1: Compute the necessary left and right padding for each task
        # Use the ordering of tasks in the padding,
        # i.e., if we have 3 tasks, task 0 will be padded to the right only, task 1 to the left and to the right,
        # task 2 to the left only
        # Tasks is an OrderedDict, so keys will also stick to this order
        task_names = list(self.tasks.keys())
        num_tasks = len(task_names)
        self.tasks2padding_of_per_head_logits = nn.ModuleDict()
        # Predictions need different padding, as in single-label tasks, the prediction for an instance is a scalar
        # instead of a vector
        self.tasks2padding_of_per_head_preds = nn.ModuleDict()

        # Also store left and right padding of task predictions -
        # DataLoaders need this to compile correctly padded targets, Trainer needs it for selecting the correct
        # predictions/targets per task
        self.tasks2left_padding_of_per_head_preds = OrderedDict()
        self.tasks2right_padding_of_per_head_preds = OrderedDict()

        # Compute where per-task logits/preds start/end in the matrix with the output of the correct task-domain-modal
        # head per instance
        self.tasks2per_head_logits_span = OrderedDict()
        self.tasks2per_head_preds_span = OrderedDict()

        # Compute index at which logits/preds start for a specific task in the 3D all-logits/preds tensor
        # (shape: (H, batch_size, width of padded logits/preds), where H = sum(t=1 to T (d_t*m_t))
        # where
        # {1,...,T}: tasks,
        # d_t: # domains of task t in model with per-domain heads, 1 otherwise,
        # m_t: # modals in task t in model with per-modal heads, 1 otherwise)
        self.tasks2start_ix_in_all_heads_tensor = OrderedDict()

        for i in range(num_tasks):

            task_name = task_names[i]
            i_task = self.tasks[task_name]
            output_vocab_length = len(i_task.output_vocab)

            # Iterate over preceding tasks
            logits_pad_left = 0
            preds_pad_left = 0
            start_ix_heads = 0

            for j in range(i):
                j_name = task_names[j]
                j_task = self.tasks[j_name]
                j_voc_length = len(j_task.output_vocab)

                logits_pad_left += j_voc_length
                preds_pad_left += 1 if j_task.single_label_task else j_voc_length
                d_t = len(j_task.target_domains) if self.use_per_domain_heads else 1
                m_t = len(j_task.target_modals) if self.use_per_modal_heads else 1
                start_ix_heads += d_t * m_t

            self.tasks2start_ix_in_all_heads_tensor[task_name] = start_ix_heads

            # Iterate over succeeding tasks
            logits_pad_right = 0
            preds_pad_right = 0

            for j in range(i+1, num_tasks):
                j_name = task_names[j]
                j_task = self.tasks[j_name]
                j_voc_length = len(j_task.output_vocab)

                logits_pad_right += j_voc_length
                preds_pad_right += 1 if j_task.single_label_task else j_voc_length

            # Store padding of logits of (full label set) heads
            # E.g., logits for an instance of MIST in the multi-task setup with MASC (task order: MASC - MIST)
            # before padding: (l1, l2, l3, l4, l5, l6)
            # after padding: (0, 0, 0, l1, l2, l3, l4, l5, l6)
            # (MASC has 3 labels)
            self.tasks2padding_of_per_head_logits[task_name] = nn.ConstantPad1d((logits_pad_left, logits_pad_right),
                                                                                0)
            self.tasks2per_head_logits_span[task_name] = (logits_pad_left, logits_pad_left + output_vocab_length)

            # Store the width of (full label set) head logits matrices
            self.per_heads_padded_logits_width = logits_pad_left + output_vocab_length + logits_pad_right

            # Store padding of predictions of (full label set) heads
            # E.g., predictions for an instance of MIST in the multi-task setup with MASC (task order: MASC - MIST)
            # before padding: (p1, p2, p3, p4, p5, p6)
            # after padding: (-100, p1, p2, p3, p4, p5, p6)
            # (MASC is a single-label dataset)
            self.tasks2padding_of_per_head_preds[task_name] = nn.ConstantPad1d((preds_pad_left, preds_pad_right),
                                                                                -100)
            self.tasks2left_padding_of_per_head_preds[task_name] = preds_pad_left
            self.tasks2right_padding_of_per_head_preds[task_name] = preds_pad_right

            # Store the width of (full label set) head predictions matrices
            preds_padding = preds_pad_left + preds_pad_right
            preds_width = 1 if self.tasks[task_name].single_label_task else output_vocab_length
            self.per_head_padded_preds_width = preds_width + preds_padding
            self.tasks2per_head_preds_span[task_name] = (preds_pad_left, preds_pad_left + preds_width)

        # Use padding of logits for confidences, as for single-label tasks, we also want to store the confidences
        # associated with the non-predicted classes
        # (Define aliases here to avoid having to remember to use the logits padding everywhere I pad confidences)
        self.tasks2padding_of_per_head_confs = self.tasks2padding_of_per_head_logits
        self.tasks2per_head_confs_span = self.tasks2per_head_logits_span

    @classmethod
    def from_args_dict(cls, args_dict: Dict, tasks: OrderedDictType[str, ModalTask], repo_path: Optional[Path]=None):
        # ----------------------Step 1: Get the embedder----------------------------------------------------------------
        embedder_type = args_dict["embedder"]["type"]
        embedder_args = args_dict["embedder"]["args"]
        embedder = getattr(embedders, embedder_type).from_args_dict(embedder_args, repo_path=repo_path)

        # ----------------------Step 2: Get the non-task output vocabs--------------------------------------------------
        non_task_output_vocabs = dict()

        output_vocabs_args = args_dict["output_vocabs"]
        for output_vocab_key in output_vocabs_args:
            ov_args = output_vocabs_args[output_vocab_key]
            output_vocab_type = ov_args["type"]
            if output_vocab_type == "BasicVocab":
                inner_args = ov_args["args"]
                vocab_path = prepend_optional_path(inner_args["vocab_filename"], repo_path)
                output_vocab = BasicVocab(vocab_path)
            elif output_vocab_type == "unary":
                output_vocab = UNARY_VOCAB
            else:
                raise NotImplementedError(f"Output vocab type {output_vocab_type} not implemented for {cls}")

            non_task_output_vocabs[output_vocab_key] = output_vocab

        # ----------------------Step 3: Set up the heads----------------------------------------------------------------
        use_per_modal_heads = args_dict["use_per_modal_heads"]
        use_per_domain_heads = args_dict["use_per_domain_heads"]
        heads_args = args_dict["heads"]
        # Input dim for heads is output dim of embedder
        heads_dict = nn.ModuleDict()

        for task_name, head_config in heads_args.items():
            task_heads = nn.ModuleDict()
            task = tasks[task_name]
            domains = task.target_domains
            modals = task.target_modals
            task_output_length = len(task.output_vocab)

            # Set up ModuleDicts and ModuleLists according to whether per-domain and/or per-modal heads should be used
            if use_per_domain_heads:
                for domain in domains:
                    if use_per_modal_heads:
                        domain_heads = nn.ModuleDict()
                        for modal in modals:
                            domain_heads[modal] = nn.ModuleList()
                    else:
                        domain_heads = nn.ModuleList()
                    task_heads[domain] = domain_heads
            else:
                if use_per_modal_heads:
                    for modal in modals:
                        task_heads[modal] = nn.ModuleList()
                else:
                    # If we use neither per-modal nor per-domain heads, make a list of task heads instead of a
                    # dictionary (list for per-label heads)
                    task_heads = nn.ModuleList()

            # Make the heads
            if "output_vocab" in head_config.keys():
                # There is an output vocab specified other than the task output vocab
                # (Currently: for per-label heads in multi-label case)
                if task.single_label_task:
                    raise ValueError(f"Unexpected input: output vocab other than the task output vocab in head config "
                                     f"for single-label task {task.name}")

                out_voc_len = len(non_task_output_vocabs[head_config["output_vocab"]])

                if use_per_domain_heads:
                    for domain in domains:
                        domain_heads = task_heads[domain]
                        if use_per_modal_heads:
                            for modal in modals:
                                for _ in range(task_output_length):
                                    domain_heads[modal].append(Head.from_args_dict(head_config, embedder.output_dim,
                                                                                   out_voc_len))
                        else:
                            for _ in range(task_output_length):
                                domain_heads.append(Head.from_args_dict(head_config, embedder.output_dim, out_voc_len))
                else:
                    if use_per_modal_heads:
                        for modal in modals:
                            for _ in range(task_output_length):
                                task_heads[modal].append(Head.from_args_dict(head_config, embedder.output_dim,
                                                                             out_voc_len))
                    else:
                        for _ in range(task_output_length):
                            task_heads.append(Head.from_args_dict(head_config, embedder.output_dim, out_voc_len))
            else:
                # No other output vocab specified
                if use_per_domain_heads:
                    for domain in domains:
                        domain_heads = task_heads[domain]
                        if use_per_modal_heads:
                            for modal in modals:
                                domain_heads[modal].append(Head.from_args_dict(head_config, embedder.output_dim,
                                                                               task_output_length))
                        else:
                            domain_heads.append(Head.from_args_dict(head_config, embedder.output_dim,
                                                                    task_output_length))
                else:
                    if use_per_modal_heads:
                        for modal in modals:
                            task_heads[modal].append(Head.from_args_dict(head_config, embedder.output_dim,
                                                                         task_output_length))
                    else:
                        task_heads.append(Head.from_args_dict(head_config, embedder.output_dim, task_output_length))

            # Add task heads to overall heads dict
            heads_dict[task_name] = task_heads

        # ----------------------Step 4: Get the loss functions----------------------------------------------------------
        loss_args = args_dict["loss"]
        single_label_loss = loss_args["single_label"]
        multi_label_loss = loss_args["multi_label"]

        return cls(embedder, heads_dict, non_task_output_vocabs, tasks, single_label_loss, multi_label_loss,
                   use_per_modal_heads, use_per_domain_heads)

    def forward(self, sentence_batch: List[SentenceWithModalSense], task_names: List[str], modal_indices: List[int],
                domain_indices: List[int], device: torch.device, mode: str = "evaluation",
                targets: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns predictions and loss if training or validating, predictions and confidences only if evaluating"""
        if mode == "training":
            assert self.training
            assert targets is not None
        elif mode == "validation":
            assert not self.training
            assert targets is not None
        else:
            assert not self.training

        compute_loss = mode == "training" or mode == "validation"

        # Embed the instances
        embedded = self.embedder(sentence_batch, device)

        # Concatenate outputs from all heads
        head_logits = []
        head_preds = []
        head_confs = []

        for task_name, task in self.tasks.items():
            domains = task.target_domains
            modals = task.target_modals
            task_heads = self.heads[task_name]
            single_labeling = task.single_label_task

            if self.use_per_domain_heads:
                for domain in domains:
                    domain_heads = task_heads[domain]

                    if self.use_per_modal_heads:
                        # domain_heads is a ModuleDict, containing ModuleLists as values
                        head_logits, head_preds, head_confs = self._forward_through_heads_dict(embedded, domain_heads,
                                                                                   single_labeling, task_name,
                                                                                   head_logits, head_preds)
                    else:
                        # domain_heads is a ModuleList
                        _logits, _preds, _confs = self._forward_through_heads_list(embedded, domain_heads, single_labeling)
                        # Append the padded matrices to the list of logits/predictions/confidences
                        head_logits.append(self.tasks2padding_of_per_head_logits[task_name](_logits))
                        head_preds.append(self.tasks2padding_of_per_head_preds[task_name](_preds))
                        head_confs.append(self.tasks2padding_of_per_head_confs[task_name](_confs))
            else:
                if self.use_per_modal_heads:
                    # task_heads is a ModuleDict, containing ModuleLists as values
                    head_logits, head_preds, head_confs = self._forward_through_heads_dict(embedded, task_heads, single_labeling,
                                                                               task_name, head_logits, head_preds, head_confs)
                else:
                    # task_heads is a ModuleList
                    _logits, _preds, _confs = self._forward_through_heads_list(embedded, task_heads, single_labeling)
                    # Append the padded matrices to the list of logits/predictions/confidences
                    head_logits.append(self.tasks2padding_of_per_head_logits[task_name](_logits))
                    head_preds.append(self.tasks2padding_of_per_head_preds[task_name](_preds))
                    head_confs.append(self.tasks2padding_of_per_head_confs[task_name](_confs))

        # Stack the logits/preds - resulting shape: (H, batch size, width of padded logits/preds),
        # where H = sum(t=1 to T (d_t*m_t)),
        # where
        # {1,...,T}: tasks,
        # d_t: # domains of task t in model with per-domain heads, 1 otherwise,
        # m_t: # modals in task t in model with per-modal heads, 1 otherwise)
        # (i.e., H is the number of (full label set) heads in the model (counting per-label heads as a single head)
        all_logits = torch.stack(head_logits)
        all_preds = torch.stack(head_preds)
        all_confs = torch.stack(head_confs)

        # Get the logits/preds from the correct (per-task per-domain per-modal) head
        # Each full label set head's predictions/logits are stored in a separate matrix, i.e., need to select the
        # correct matrix from the (first dimension of the) 3D tensor
        matrix_indices = []

        for (task, d_ix, m_ix) in zip(task_names, domain_indices, modal_indices):
            # Get the index of the first matrix belonging to the task of the given instance
            task_start_in_all_heads_tensor = self.tasks2start_ix_in_all_heads_tensor[task]

            # If we use per-domain heads, we need to find the index where the head for the domain of the given instance
            # starts
            d_ix = d_ix if self.use_per_domain_heads else 0

            # If we use per-modal heads, we need to find the index where the head for the modal of the given instance
            # starts
            m_ix = m_ix if self.use_per_modal_heads else 0

            # Store how many heads we have for each modal (potentially in a given domain)
            # If we use per-modal heads: factors in only if we also use per-domain heads
            # (to leap over the correct number of per-domain per-modal heads to find the start of our domain)
            # If we do not use per-modal heads: need to set this to 1 to not cancel out d_ix in case we use per-domain
            # heads
            num_per_modal_heads = len(self.tasks[task].target_modals) if self.use_per_modal_heads else 1
            matrix_indices.append(task_start_in_all_heads_tensor + d_ix * num_per_modal_heads + m_ix)

        row_tensor = torch.arange(len(matrix_indices), device=device)
        matrix_indices = torch.tensor(matrix_indices, device=device)
        logits = all_logits[matrix_indices, row_tensor, :]  # Shape (batch_size x self.per_head_logits_width)
        predictions = all_preds[matrix_indices, row_tensor, :]  # Shape (batch_size x self.per_head_preds_width)
        confidences = all_confs[matrix_indices, row_tensor, :] # Shape (batch_size x self.per_head_logits_width)

        if compute_loss:
            losses = []
            tasks_in_batch = set(task_names)
            for task_name, loss_fn in self.tasks2losses.items():
                if task_name not in tasks_in_batch:
                    # Avoid computation of nan loss if there are no instances of the task in the current batch
                    continue
                logits_start, logits_end = self.tasks2per_head_logits_span[task_name]
                task_logits = logits[:, logits_start:logits_end]
                targets_start, targets_end = self.tasks2per_head_preds_span[task_name]
                task_targets = targets[:, targets_start:targets_end]
                if self.tasks[task_name].single_label_task:
                    task_targets = task_targets.flatten()
                task_loss = loss_fn(task_logits, task_targets)
                losses.append(task_loss)

            loss = sum(losses)
            return predictions, loss

        return predictions, confidences

    def _forward_through_heads_dict(self, embedded: torch.Tensor, heads_dict: nn.ModuleDict, single_labeling: bool,
                                    task_name: str, head_logits: List[torch.Tensor], head_preds: List[torch.Tensor],
                                     head_confs: List[torch.Tensor])\
            -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Helper function to pass an input through a dict of head lists.
        Appends the stacked logits and predictions made by these heads as well as the confidences associated with the predictions to head_logits, head_preds and head_confs respectively,
        which are then returned
        """
        # heads is a ModuleDict, containing ModuleLists as values
        # This is ordered (i.e., iteration over values respects the order of insertion,
        # so we can compute the index of a modal head)
        for head_list in heads_dict.values():
            _logits, _preds, _confs = self._forward_through_heads_list(embedded, head_list, single_labeling)

            # Append the padded matrices to the list of logits/predictions
            head_logits.append(self.tasks2padding_of_per_head_logits[task_name](_logits))
            head_preds.append(self.tasks2padding_of_per_head_preds[task_name](_preds))
            head_confs.append(self.tasks2padding_of_per_head_confs[task_name](_confs))

        return head_logits, head_preds, head_confs

    def _forward_through_heads_list(self, embedded: torch.Tensor, head_list: nn.ModuleList, single_labeling: bool)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper function to pass an input through a list of heads.
        Returns the stacked logits and predictions made by these heads as well as the confidences associated with the predictions.
        """
        if len(head_list) == 1:
            # Only one head
            head = head_list[0]
            if head.output_dim == 1:
                # Only one binary head would only make sense if we had a binary label set
                raise NotImplementedError
            _logits = head(embedded)
            head_probs = head.activation(_logits)
            _confs = head_probs
            if single_labeling:
                _preds = torch.argmax(head_probs, dim=1)  # Shape: batch_size
                # Unsqueeze to shape (batch_size, 1) for later stacking of the task matrix with other tasks'
                # matrices
                _preds = _preds.unsqueeze(1)
            else:
                _preds = self._get_multilabel_prediction(head_probs, head.sigmoid_threshold)

        else:
            # Multiple heads -> must be binary ones (one per label)
            heads_logits = []
            heads_preds = []
            heads_confs = []
            for head in head_list:
                if head.output_dim == 1:
                    _logits = head(embedded)
                    heads_logits.append(_logits)
                    head_probs = head.activation(_logits)
                    heads_confs.append(head_probs)
                    heads_preds.append(torch.where(head_probs > 0.5, 1.0, 0.0))
                else:
                    raise NotImplementedError

            # Get the full logits/preds/confidences matrix
            _logits = torch.hstack(heads_logits)
            _preds = torch.hstack(heads_preds)
            _confs = torch.hstack(heads_confs)

        return _logits, _preds, _confs

    @staticmethod
    def _get_multilabel_prediction(probabilities: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Helper function to get a tensor encoding predicted labels from a tensor with label probabilities.
        If no probability is above the threshold, the label with the largest probability will be predicted.
        """
        return torch.where((probabilities > threshold) | (probabilities == torch.max(probabilities, dim=1, keepdim=True)[0]), 1.0, 0.0)  # [0] because torch.max returns a tuple of values and indices
