# coding=utf-8
# Copyright 2021 The Tensor2Tensor Authors.
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

"""Imports for problem modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import six
from six.moves import range  # pylint: disable=redefined-builtin

MODULES = [
    "TensorFlow.nlp.transformer.data_generators.translate_encs_cubbitt",
    "TensorFlow.nlp.transformer.data_generators.translate_encs",
    "TensorFlow.nlp.transformer.data_generators.translate_ende",
    "TensorFlow.nlp.transformer.data_generators.translate_enes",
    "TensorFlow.nlp.transformer.data_generators.translate_enet",
    "TensorFlow.nlp.transformer.data_generators.translate_enfr",
    "TensorFlow.nlp.transformer.data_generators.translate_enid",
    "TensorFlow.nlp.transformer.data_generators.translate_enmk",
    "TensorFlow.nlp.transformer.data_generators.translate_envi",
    "TensorFlow.nlp.transformer.data_generators.translate_enzh",
]
ALL_MODULES = list(MODULES)



def _is_import_err_msg(err_str, module):
  parts = module.split(".")
  suffixes = [".".join(parts[i:]) for i in range(len(parts))]
  prefixes = [".".join(parts[:i]) for i in range(len(parts))]
  return err_str in (["No module named %s" % suffix for suffix in suffixes] +
                     ["No module named '%s'" % suffix for suffix in suffixes] +
                     ["No module named %s" % prefix for prefix in prefixes] +
                     ["No module named '%s'" % prefix for prefix in prefixes])


def _handle_errors(errors):
  """Log out and possibly reraise errors during import."""
  if not errors:
    return
  log_all = True  # pylint: disable=unused-variable
  err_msg = "T2T: skipped importing {num_missing} data_generators modules."
  print(err_msg.format(num_missing=len(errors)))
  for module, err in errors:
    err_str = str(err)
    if log_all:
      print("Did not import module: %s; Cause: %s" % (module, err_str))
    if not _is_import_err_msg(err_str, module):
      print("From module %s" % module)
      raise err


def import_modules(modules):
  errors = []
  for module in modules:
    try:
      importlib.import_module(module)
    except ImportError as error:
      errors.append((module, error))
  _handle_errors(errors)
