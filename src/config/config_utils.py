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
# Authors: Stefan Grünewald, Sophie Henning

import json
from collections import OrderedDict
from functools import reduce
from operator import getitem
from pathlib import Path
from math import ceil


def read_json(fname):
    if isinstance(fname, str):
        fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def set_by_path(tree, keys, value) -> None:
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split('.')
    penultimate_level = get_by_path(tree, keys[:-1])
    penultimate_level[keys[-1]] = value


def get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def compute_num_epochs_for_given_num_steps(num_steps: int, batch_size: int, num_sentences: int):
    """Calculate for how many epochs we need to train the classifier to reach a given number of steps"""
    num_batches = ceil(num_sentences / batch_size)

    return ceil(num_steps / num_batches)
