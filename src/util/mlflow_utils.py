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

import os
import mlflow
from pathlib import Path


def get_smallest_free_experiment_id(mlruns_folder: Path) -> int:
    # List all immediate subfolder names using integers as names
    list_mlruns_integer_subfolders = []
    for f in os.scandir(mlruns_folder):
        if f.is_dir():
            try:
                name_as_int = int(f.name)
                list_mlruns_integer_subfolders.append(name_as_int)
            except ValueError:
                # Ignore non-integer folder names
                continue
    if not list_mlruns_integer_subfolders:
        # Empty list
        return 0
    else:
        # Sort list (descending)
        list_mlruns_integer_subfolders.sort(reverse=True)
        return list_mlruns_integer_subfolders[0] + 1


def create_mlflow_experiment(mlruns_folder: Path, name: str) -> str:
    mlflow.set_tracking_uri(f"file://{mlruns_folder}")
    experiment_name = f"{name}{get_smallest_free_experiment_id(mlruns_folder)}"

    # Create an experiment for nicer UI
    return mlflow.create_experiment(experiment_name)