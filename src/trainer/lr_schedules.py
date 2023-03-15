# This source code is from the STEPS Parser (w/ adaptations by Stefan Grünewald)
#   (https://github.com/boschresearch/steps-parser/blob/master/src/trainer/lr_scheduler.py)
# Copyright (c) 2020 Robert Bosch GmbH
# This source code is licensed under the AGPL v3 license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.
# Author: Stefan Grünewald

import math
from torch.optim.lr_scheduler import LambdaLR


class WarmupSchedule:
    """Wrapper for LR schedule with warmup."""
    def __init__(self, warmup_steps):
        """
        Args:
            warmup_steps: Number of steps for linear warmup. One step = one batch
        """
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step == 0:
            return 0
        elif 1 <= step <= self.warmup_steps:
            return step / self.warmup_steps
        else:
            return 1


class TriangularSchedule:
    """Wrapper for LR schedule with linear warmup and linear decay."""
    def __init__(self, warmup_steps, decay_steps):
        """
        Args:
            warmup_steps: Number of steps for linear warmup.
        """
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        if step == 0:
            return 0
        elif 1 <= step <= self.warmup_steps:
            return step / self.warmup_steps
        else:
            return max(0, 1-(step-self.warmup_steps)/self.decay_steps)


class SqrtSchedule:
    """Wrapper for Noam LR schedule."""
    def __init__(self, warmup_steps):
        """
        Args:
            warmup_steps: Number of steps for linear warmup.
        """
        self.warmup_steps = warmup_steps
        self.sqrt_warmup_steps = warmup_steps**0.5
        self.inv_warmup_steps = warmup_steps**(-1.5)

    def __call__(self, step):
        if step == 0:
            return 0
        else:
            return self.sqrt_warmup_steps * min(step**(-0.5), step*self.inv_warmup_steps)


class WarmRestartSchedule:
    """Wrapper for cosine annealing with warmup and warm restarts."""
    def __init__(self, warmup_steps, T_0, T_mult=1, eta_min=0):
        """
        Args:
            warmup_steps: Number of linear warmup steps.
            T_0: Initial cycle length.
            T_mult: Cycle length growth factor.
            eta_min: Minimum learning rate scaling factor.
        """
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

    def __call__(self, step):
        if step <= self.warmup_steps:
            return step / self.warmup_steps
        else:
            step = step - self.warmup_steps

            # Determine current cycle
            if step >= self.T_0:
                if self.T_mult == 1:
                    T_cur = step % self.T_0
                    T_i = self.T_0
                else:
                    n = int(math.log((step / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    T_cur = step - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    T_i = self.T_0 * self.T_mult ** (n)
            else:
                T_i = self.T_0
                T_cur = step

        return self.eta_min + (1 - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2