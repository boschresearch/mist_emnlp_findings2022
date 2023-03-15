# This source code is from the STEPS Parser (w/ adaptations by Sophie Henning)
#   (https://github.com/boschresearch/steps-parser/blob/master/src/trainer/losses.py)
# Copyright (c) 2020 Robert Bosch GmbH
# This source code is licensed under the AGPL v3 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
# Author: Stefan Grünewald

import torch

from torch.nn.modules.loss import _Loss
from torch.nn import CrossEntropyLoss  # import needed because the config parser searches for losses in this file
from torch.nn import BCEWithLogitsLoss


class BCEWithLogitsLossWithIgnore(_Loss):
    """Custom BCEWithLogitsLoss that ignores indices where target tensor is negative.
    Useful when working with padding.

    Additionally, makes BCE loss work with integer targets.
    """

    def __init__(self):
        super(BCEWithLogitsLossWithIgnore, self).__init__()
        self.bce_with_logits_loss = BCEWithLogitsLoss()

    def forward(self, input, target):
        assert input.shape == target.shape

        # Get the non-ignored inputs/targets
        input_non_ignored = input[target >= 0]
        target_non_ignored = target[target >= 0]

        # Masked selection returns an 1-D tensor -> need to reshape it to have output_shape columns again
        # Number of task instances in the batch is inferred by reshape
        input_non_ignored = input_non_ignored.reshape(-1, target.shape[1])
        target_non_ignored = target_non_ignored.reshape(-1, target.shape[1])

        assert input_non_ignored.shape == target_non_ignored.shape

        if target_non_ignored.dtype != torch.float32:
            target_non_ignored = target_non_ignored.float()

        return self.bce_with_logits_loss(input_non_ignored, target_non_ignored)