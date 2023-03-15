# This source code is from the STEPS Parser (w/ heavy adaptations by Sophie Henning)
#   (https://github.com/boschresearch/steps-parser/blob/master/src/data_handling/custom_conll_dataset.py)
# Copyright (c) 2020 Robert Bosch GmbH
# This source code is licensed under the AGPL v3 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
# Author: Stefan Grünewald

from torch.utils.data import Dataset
from typing import List, TextIO, Iterator, Optional, Dict, OrderedDict

from .sentence_with_modal_sense import SentenceWithModalSense
from .constants import MULTI_LABEL_INFO, SINGLE_LABEL_INFO
from .tasks import ModalTask, NO_DOMAINS


class SentenceDataset(Dataset):
    """
    An object of this class represents a (map-style) dataset of sentences annotated with a modal sense.
    """
    def __init__(self, single_labeling: Optional[bool] = None):
        self.sentences: List[SentenceWithModalSense] = []
        if single_labeling is not None:
            self.single_labeling = single_labeling

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, item) -> SentenceWithModalSense:
        return self.sentences[item]

    def append_sentence(self, sent: SentenceWithModalSense) -> None:
        self.sentences.append(sent)

    @staticmethod
    def check_sentence(sent: SentenceWithModalSense, task: ModalTask) -> bool:
        """
        Helper method to check whether a sentence should be added to the data set, according to the specification of
        its corresponding task.
        """
        if sent.target_modal not in task.target_modals:
            return False
        if task.target_domains != NO_DOMAINS:
            # Task has domains
            if sent.domain not in task.target_domains:
                return False

        return True

    @staticmethod
    def from_corpus_files(corpus_file_names: Dict[str, str], tasks: OrderedDict[str, ModalTask]):
        """
        Read in a dataset from corpus files in sentence-wise format.
        This method is used in src.

        Args:
            corpus_file_names: maps task names to corpus file paths
            tasks: OrderedDict from task names to ModalTasks

        Returns:
            A SentenceDataset object containing the sentences in the input corpus files as SentenceWithModalSense objects.
        """
        dataset = SentenceDataset()
        for task_name, corpus_path in corpus_file_names.items():
            first = True
            task = tasks[task_name]

            with open(corpus_path, "r", encoding="utf-8") as corpus:
                for raw_sent in _iter_corpus_sentences(corpus):
                    if first:
                        # Very first line contains info whether problem is single- or multi-label
                        info = raw_sent[0]
                        # Check if task description matches the data provided
                        if info == MULTI_LABEL_INFO:
                            if task.single_label_task:
                                raise ValueError(f"Providing multi-label data for single-label task {task_name}")
                        elif info == SINGLE_LABEL_INFO:
                            if not task.single_label_task:
                                raise ValueError(f"Providing single-label data for multi-label task {task_name}")
                        else:
                            raise ValueError(f"Unexpected first corpus line {info}")
                        first = False
                        continue

                    # Real sentence
                    processed_sent = SentenceWithModalSense.from_corpus(raw_sent, task_name)

                    # Add sentence to data set if it satisfies the conditions specified in the config files
                    if SentenceDataset.check_sentence(processed_sent, task):
                        dataset.append_sentence(processed_sent)

        return dataset

    @staticmethod
    def from_corpus_file(corpus_file_name: str, task_name: str, target_modals: List[str],
                         target_domains: Optional[List[str]] = None):
        """
        Read in a dataset from a corpus file in sentence-wise format.
        This method is used when processing corpora, e.g., for generating k-fold CV splits.

        Args:
            corpus_file_name: Path to the corpus file to read from.
            target_modals: Only sentences whose target modal is an element of this list will be appended to the dataset.
            target_domains: If this is not None, only sentences from target_domains will be added to the dataset.

        Returns:
            A SentenceDataset object containing the sentences in the input corpus file with their labels.

        """
        dataset = SentenceDataset(single_labeling=True)
        first = True

        with open(corpus_file_name, "r", encoding="utf-8") as corpus:
            for raw_sent in _iter_corpus_sentences(corpus):
                if first:
                    # Very first line contains info whether problem is single- or multi-label
                    info = raw_sent[0]
                    if info == MULTI_LABEL_INFO:
                        dataset.single_labeling = False
                    elif info != SINGLE_LABEL_INFO:
                        raise ValueError(f"Unexpected first corpus line {info}")
                    first = False
                    continue
                processed_sent = SentenceWithModalSense.from_corpus(raw_sent, task_name)
                if (processed_sent.target_modal in target_modals) \
                        and (target_domains is None or processed_sent.domain in target_domains):
                    dataset.append_sentence(processed_sent)

        return dataset


# The following function is adapted from pyconll
# (https://github.com/pyconll/pyconll/blob/master/pyconll/_parser.py)
# Copyright 2018 Matias Grioni, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
def _iter_corpus_sentences(corpus: TextIO) -> Iterator[List[str]]:
    """
    Helper function to iterate over the corpus data in the given file stream.

    Args:
        corpus: stream from corpus file

    Yields:
        An iterator over the corpus lines for each sentence.
    """
    sent_lines = []
    for line in corpus:
        line = line.strip()

        # Sentences are separated by an empty lne
        if line:
            sent_lines.append(line)
        else:
            # Line was empty -> previous sentence ended
            if sent_lines:
                yield sent_lines
                sent_lines = []

    if sent_lines:
        yield sent_lines

