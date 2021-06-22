# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from absl import flags
from absl import logging
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.framework import op_callbacks
from tensorflow.python.ops import gen_debug_ops
import tensorflow as tf
import re
import os
import json
from TensorFlow.common.horovod_helpers import horovod_enabled, hvd_rank


flags.DEFINE_string(name='dump_config', default=None,
                    help='Defines config for tensor dumping')


class _DumpCallback(object):
    def __init__(self, dump_root, tensor_debug_mode, circular_buffer_size, op_regex):
        self._dump_root = dump_root
        if horovod_enabled():
            self._dump_root = os.path.join(
                self._dump_root, f"rank_{hvd_rank()}")
        self._tensor_debug_mode = debug_event_pb2.TensorDebugMode.Value(
            tensor_debug_mode)
        self._circular_buffer_size = circular_buffer_size
        self._op_regex = re.compile(op_regex) if isinstance(
            op_regex, str) else op_regex
        self._tfdbg_run_id = ''
        self._dump_op_counter = 0

        debug_writer_args = {
            "dump_root": self._dump_root,
            "circular_buffer_size": self._circular_buffer_size
        }

        if not tf.__version__.startswith("2.2"):
            debug_writer_args["tfdbg_run_id"] = self._tfdbg_run_id

        self._writer = debug_events_writer.DebugEventsWriter(
            **debug_writer_args)

    def callback(self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
        if op_name is not None and self._op_regex.match(op_name):
            graph_name = "missing-graph-name"
            if graph is not None and hasattr(graph, "name"):
                graph_name = graph.name

            logging.info("Adding dump op for '%s' of type '%s' from graph '%s'" % (
                op_name, op_type, graph_name))

            new_outputs = []

            for output_slot, output in enumerate(outputs):
                debug_identity_op_kwargs = {
                    "tfdbg_context_id": graph_name,
                    "op_name": op_name,
                    "output_slot": output_slot,
                    "tensor_debug_mode": self._tensor_debug_mode,
                    "debug_urls": ["file://%s" % self._dump_root],
                    "name": "dump_%d" % self._dump_op_counter
                }

                if not tf.__version__.startswith("2.2"):
                    debug_identity_op_kwargs["circular_buffer_size"] = self._circular_buffer_size
                    debug_identity_op_kwargs["tfdbg_run_id"] = self._tfdbg_run_id

                self._dump_op_counter = self._dump_op_counter + 1
                new_outputs.append(gen_debug_ops.debug_identity_v2(
                    output, **debug_identity_op_kwargs))

            return new_outputs
        else:
            return None

    def __enter__(self, *args, **kwargs):
        op_callbacks.add_op_callback(self.callback)
        logging.info("Enabled tensor dumping")

    def __exit__(self, *args, **kwargs):
        op_callbacks.remove_op_callback(self.callback)
        logging.info("Disabled tensor dumping")

    def __del__(self):
        self._writer.Close()


class _Dummy(object):
    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def dump_callback(config_file=None):
    if config_file is not None:
        kwargs = json.load(open(config_file, 'r'))
        return _DumpCallback(**kwargs)
    try:
        kwargs = json.load(open(flags.FLAGS.dump_config, 'r'))
        return _DumpCallback(**kwargs)
    except:
        return _Dummy()
