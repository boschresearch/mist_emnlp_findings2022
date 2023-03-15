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

from typing import List, Optional, Tuple, Dict, OrderedDict

from .vocab import BasicVocab

from .tasks import ModalTask, NO_DOMAINS


class SentenceWithModalSense:
    """
    A sentence labeled with a modal sense. The label(s) refer to the target modal.
    """
    def __init__(self, tokens: List[str], labels: List[str], sent_id: str, target_modal: str,
                 target_modal_position: int, task: str, domain: Optional[str] = None, doc: Optional[str] = None):
        self.tokens = tokens
        self.labels = labels
        self.target_modal = target_modal
        self.target_modal_position = target_modal_position  # index starts with 0
        self.task = task
        self.domain = NO_DOMAINS[0] if domain is None else domain
        self.doc = doc
        self.id = sent_id

    def __len__(self):
        """Return the number of tokens in this sentence"""
        return len(self.tokens)

    def __str__(self):
        """Return the tokens of the sentence as a string."""
        return f"SentenceWithModalSense({' '.join(self.tokens)})"

    def has_multiple_labels(self) -> bool:
        return len(self.labels) > 1

    def get_labels_as_str(self) -> str:
        return "-".join(sorted(self.labels))

    def to_corpus(self) -> str:
        """Returns a string in the format of the method from_corpus."""
        start = f"#{self.domain}\t{self.doc}\t" if self.domain and self.doc else ""
        start = f"{start}{self.id}\n"
        token_string = "\n".join(self.tokens)

        line = f"{start}" \
               f"{self.target_modal}\n" \
               f"{self.target_modal_position}\n" \
               f"{self.get_labels_as_str()}\n" \
               f"{token_string}"

        return line

    @staticmethod
    def from_corpus(corpus_lines: List[str], task: str):
        """
        Create a SentenceWithModalSense from an iterable of corpus lines.

        Args:
            corpus_lines: List of strings representing the sentence in sentence-wise format
                Format [optional]:
                #[Subcorpus\tDoc\t]sentence ID
                Target modal
                Position of target modal in token list (indexed by 0)
                Label (multi-label classification: labels separated by commas)
                Token_0
                ...
                Token_n-1
                blank line between sentences
            task: name of the task the corpus lines are from

        Returns:
            A SentenceWithModalSense object representing the sentence specified in the corpus data.
        """
        # Read in meta-data if there is some
        domain, doc, sent_id = None, None, None
        if corpus_lines[0].startswith("#"):
            meta_data = corpus_lines[0].lstrip("#")
            split = meta_data.split("\t")
            if len(split) == 3:
                domain, doc, sent_id = split[0], split[1], split[2]
            elif len(split) == 1:
                sent_id = split[0]
            else:
                raise ValueError(f"Unexpected format of meta-data line: {meta_data}")
            corpus_lines = corpus_lines[1:]
        else:
            raise ValueError("Missing meta-data line")

        assert len(corpus_lines) > 3  # A sentence has to contain at least one token

        target_modal = corpus_lines[0]
        target_modal_position = int(corpus_lines[1])
        labels = corpus_lines[2].split("-")
        token_lines = corpus_lines[3:]
        tokens = []

        for token_line in token_lines:
            split = token_line.split("\t")
            tokens.append(split[0])
            assert len(split) == 1

        return SentenceWithModalSense(tokens, labels, sent_id, target_modal, target_modal_position, task, domain, doc)


def encode_labels(labels: List[str], vocab: BasicVocab, single_label: bool) -> List[int]:
    """Encode list of labels using vocab, using one-hot encoding in single-label case and multi-hot encoding in
    multi-label case"""
    if single_label:
        encoded_labels = [vocab.token2ix(labels[0])]
    else:
        applying_labels = {vocab.token2ix(label) for label in labels}
        encoded_labels = [1 if i in applying_labels else 0 for i in range(len(vocab))]
    return encoded_labels


def get_tensorized_labels(sentences: List[SentenceWithModalSense], tasks: OrderedDict[str, ModalTask],
                          left_padding: Dict[str, int], right_padding: Dict[str, int]) \
        -> Tuple[torch.Tensor, List[str], List[int], List[int]]:
    """
    For an iterable of SentenceWithModalSense objects, create a batched index tensor for the labels, a task mask,
    a target modal mask, and a target domain mask.

    Output is a tuple of:
    a) a tensor of shape num_sentences if `single_label_classification=True`, else (num_sentences, label_vocab_length)
    b) a list of strings of length num_sentences, position i: name of the task to which sentence i belongs
    c) a list of integers of length num_sentences which at position i contains the index of sentence i's target modal
    in target_modals
    d) a list of integers of length num_sentences which at position i contains the index of sentence i's domain
    (subcorpus) in target_domains if sentences are from a corpus with domains, 0 at every position otherwise
    """
    label_indices = []
    task_names = []
    modal_indices = []
    domain_indices = []

    for sentence in sentences:
        # Get task
        task_name = sentence.task
        task_names.append(task_name)
        task = tasks[task_name]

        # Encode labels
        encoded_labels = encode_labels(sentence.labels, task.output_vocab, task.single_label_task)

        # Pad targets with -100 for other tasks' target cells (-100: CrossEntropyLoss' default ignore index)
        left = [-100 for _ in range(left_padding[task_name])]
        right = [-100 for _ in range(right_padding[task_name])]
        tgts = left + encoded_labels + right
        label_indices.append(tgts)

        # Get target modal
        modal_indices.append(task.target_modals.index(sentence.target_modal))

        # Get target domain
        domain_indices.append(task.target_domains.index(sentence.domain))

    label_tensor = torch.LongTensor(label_indices)

    return label_tensor, task_names, modal_indices, domain_indices


