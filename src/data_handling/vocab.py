# This source code is from the STEPS Parser (w/ adaptations by Sophie Henning)
#   (https://github.com/boschresearch/steps-parser/blob/master/src/data_handling/vocab.py)
# Copyright (c) 2020 Robert Bosch GmbH
# This source code is licensed under the AGPL v3 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
# Author: Stefan Grünewald

from typing import Dict, Optional, Set


class BasicVocab:
    """Class for mapping labels/tokens to indices and vice versa."""

    def __init__(self, vocab_filename=None, ignore_label="__IGNORE__", ignore_index=-1,
                 unknown_token_string: Optional[str] = None):
        """A vocabulary is read from a file in which each label constitutes one line. The index associated with each
        label is the index of the line that label occurred in (counting from 0).

        In addition, a special label and index (`ignore_label` and `ignore_index`) are added to signify content which
        should be ignored in parsing/tagging tasks.

        Args:
            vocab_filename: Name of the file to read the vocabulary from.
            ignore_label: Special label signifying ignored content. Default: `__IGNORE__`.
            ignore_index: Special index signifying ignored content. Should be negative to avoid collisions with "true"
              indices. Default: `-1`.
            unknown_token_string: Special token for unknown tokens. Will be added at the "end" of the vocab (i.e.,
            lowest available index).
        """
        self.ix2token_data = dict()
        self.token2ix_data = dict()

        self.vocab_filename = vocab_filename

        if self.vocab_filename is not None:
            with open(vocab_filename) as vocab_file:
                for ix, line in enumerate(vocab_file):
                    token = line.strip()

                    self.ix2token_data[ix] = token
                    self.token2ix_data[token] = ix

        self.ignore_label = ignore_label
        self.ignore_index = ignore_index

        self.ix2token_data[ignore_index] = ignore_label
        self.token2ix_data[ignore_label] = ignore_index

        self.unknown_token_string = unknown_token_string

        if unknown_token_string:
            # Add a special token for unknown tokens
            next_index = len(self)
            self.token2ix_data[unknown_token_string] = next_index
            self.ix2token_data[next_index] = unknown_token_string

        assert self.is_consistent()

    @staticmethod
    def from_token2ix_dict(token2ix: Dict[str, int], ignore_label: str, ignore_index: int,
                           unknown_token_string: Optional[str] = None):
        """
        Generate a BasicVocab object from an already existing token-to-index dictionary.

        Args:
            token2ix: Dictionary mapping tokens to indices, assumed to already contain an entry for an ignore label and
            for the unknown token if unknown_token_string is passed as an argument
            ignore_label: Ignore label used in token2ix
            ignore_index: Ignore index used in token2ix
            unknown_token_string: Special token for unknown tokens used in token2ix

        Returns:
            A BasicVocab object with token2ix's mapping.
        """
        vocab = BasicVocab()
        vocab.ignore_index = ignore_index
        vocab.ignore_label = ignore_label
        vocab.token2ix_data = token2ix
        vocab.ix2token_data = dict()

        for (token, index) in token2ix.items():
            vocab.ix2token_data[index] = token

        if unknown_token_string:
            # Assuming the unknown token string was already contained n token2ix
            vocab.unknown_token_string = unknown_token_string

        assert vocab.is_consistent()

        return vocab

    def __len__(self):
        return len(self.ix2token_data) - 1  # Do not count built-in "ignore" label

    def __str__(self):
        # Do not consider built-in "ignore" label
        return "\n".join(self.ix2token_data[ix] for ix in sorted(self.ix2token_data.keys()) if ix >= 0)

    def contains(self, token: str) -> bool:
        """Check if the vocabulary contains a specific token"""
        return token in self.token2ix_data.keys()

    def ix2token(self, ix):
        """Get the token associated with index `ix`."""
        return self.ix2token_data[ix]

    def token2ix(self, token: str):
        """
        Get the index associated with token `token`.
        modal: additional argument to be used if there are invalid modal-label combinations (label = token)
        """
        return self.token2ix_data[token]

    def add(self, token):
        """Adds a token to the vocabulary if it does not already exist."""
        if token not in self.token2ix_data:
            new_ix = len(self)

            self.token2ix_data[token] = new_ix
            self.ix2token_data[new_ix] = token

    def get_real_tokens(self) -> Set[str]:
        """Return all tokens in the dictionary except for the ignore label."""
        return {token for token in self.token2ix_data.keys() if token != self.ignore_label}

    def to_file(self, vocab_filename):
        """Write vocabulary to a file."""
        with open(vocab_filename, "w") as vocab_file:
            vocab_file.write(str(self))

    def is_consistent(self):
        """Checks if all index mappings match up. Used for debugging."""
        if len(self.ix2token_data) != len(self.token2ix_data):
            return False

        try:
            for token, ix in self.token2ix_data.items():
                if self.ix2token_data[ix] != token:
                    return False
        except IndexError:
            return False

        if "[null]" in self.token2ix_data:
            assert self.token2ix_data["[null]"] == 0

        return True


UNARY_VOCAB = BasicVocab()
UNARY_VOCAB.add("true")