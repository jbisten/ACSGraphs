# -*- coding: utf-8 -*-

# Copyright (C) 2019  Andrea Vázquez Varela

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#Authors:
# Narciso López López
# Andrea Vázquez Varela
#Creation date: 30/10/2019
#Last update: 06/03/2020

import shutil
import os
import IOFibers as IOF
from pathlib import Path

def create_output(path):
    path = Path(path)
    path.mkdir(exist_ok=True)
    return path

def read_bundles(path):
    bundles, names, fiber_ids = IOF.read_bundles(path)
    return bundles[0]

def write_bundles(clusters, cluster_fiber_ids, centroids, bundles_dir, out_path):
    IOF.write_bundles(out_path+"/finalClusters.bundles",clusters)
    IOF.write_cluster_fiber_ids(cluster_fiber_ids, out_path)
    IOF.write_bundles(out_path+"/centroids.bundles",centroids)


