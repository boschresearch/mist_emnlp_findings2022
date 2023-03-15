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

import random
import numpy as np
import torch
from pathlib import Path

from typing import List, Optional


def pretty_print_ratio(numerator: int, denominator: int, decimal_positions: int = 1) -> str:
    return f"{numerator}/{denominator} = {round(numerator/denominator*100, decimal_positions)}%"


def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_general_name_of_validation_metric(tasks_in_validation_metric: List[str], validation_metric: str) -> str:
    return f"macro_of_{'_and_'.join(tasks_in_validation_metric)}_per_domain_per_modal_{validation_metric}"


def prepend_optional_path(path: Path, optional_path: Optional[Path]) -> Path:
    if optional_path is None:
        return path
    else:
        return optional_path / path