# This source code is from the STEPS Parser (w/ adaptations by Sophie Henning)
#   (https://github.com/boschresearch/steps-parser/blob/master/src/data_handling/data_loaders.py)
# Copyright (c) 2020 Robert Bosch GmbH
# This source code is licensed under the AGPL v3 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
# Author: Stefan Grünewald


import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, OrderedDict

from .sentence_dataset import SentenceDataset
from .sentence_with_modal_sense import SentenceWithModalSense, get_tensorized_labels
from .tasks import ModalTask
from modules.modal_classifier import ModalClassifier


class SentenceClassificationDataLoader(DataLoader):
    """DataLoader class for loading batches of sentences from a corpus file in sentence-wise format."""
    def __init__(self, corpus_paths: Dict[str, str], model: ModalClassifier,
                 batch_size: int = 10, shuffle=True, num_workers=1):
        # Read in dataset
        _dataset = SentenceDataset.from_corpus_files(corpus_paths, model.tasks)

        super().__init__(_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=lambda x: _batchify(x, model.tasks, model.tasks2left_padding_of_per_head_preds,
                                                        model.tasks2right_padding_of_per_head_preds))

        self.corpus_paths = corpus_paths


def _batchify(sentences: List[SentenceWithModalSense], tasks: OrderedDict[str, ModalTask], left_padding: Dict[str, int],
              right_padding: Dict[str, int]) \
        -> Tuple[List, torch.Tensor, List[str], List[int], List[int]]:
    """
    Helper function to create model input / gold output from a bunch of SentenceWithModalSenses objects.

    Output: A tuple whose first element is the list of SentenceWithModalSenses and whose second element is the
    target tensor.
    """
    label_tensors, task_names, modal_indices, domain_indices = \
        get_tensorized_labels(sentences, tasks, left_padding, right_padding)

    return sentences, label_tensors, task_names, modal_indices, domain_indices