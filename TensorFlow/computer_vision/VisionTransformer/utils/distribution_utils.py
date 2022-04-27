# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
"""Helper functions for running models in a distributed setting."""


import json
import os


def comm_size():
    return int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))


def comm_rank():
    return int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))


def configure_cluster(worker_hosts=None, task_index=-1):
    """Set multi-worker cluster spec in TF_CONFIG environment variable.

    Args:
      worker_hosts: comma-separated list of worker ip:port pairs.

    Returns:
      Number of workers in the cluster.
    """
    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
    if tf_config:
        num_workers = (len(tf_config['cluster'].get('chief', [])) +
                       len(tf_config['cluster'].get('worker', [])))
    elif worker_hosts:
        workers = worker_hosts.split(',')
        num_workers = len(workers)
        if num_workers > 1 and task_index < 0:
            raise ValueError(
                'Must specify task_index when number of workers > 1')
        task_index = 0 if num_workers == 1 else task_index
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': {
                'worker': workers
            },
            'task': {'type': 'worker', 'index': task_index}
        })
    else:
        num_workers = 1
    return num_workers
