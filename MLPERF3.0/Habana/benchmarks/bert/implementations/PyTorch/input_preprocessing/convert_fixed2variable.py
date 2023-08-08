# coding=utf-8
# Copyright (c) 2019-2022 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 MLBenchmark Group. All rights reserved.

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

"""Create masked LM/next sentence masked_lm TF examples for BERT."""


import argparse
import time
import logging
import collections
import h5py
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Eval sample picker for BERT.")
parser.add_argument(
    '--input_hdf5_file',
    type=str,
    default='',
    help='Input hdf5_file path')
parser.add_argument(
    '--output_hdf5_file',
    type=str,
    default='',
    help='Output hdf5_file path')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
  tic = time.time()
  h5_ifile = h5py.File(args.input_hdf5_file, 'r')
  num_examples = h5_ifile.get('next_sentence_labels').shape[0]

#  hdf5_compression_method = "gzip"
  hdf5_compression_method = None

  h5_writer = h5py.File(args.output_hdf5_file, 'w')
  input_ids = h5_writer.create_dataset('input_ids', (num_examples,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
  segment_ids = h5_writer.create_dataset('segment_ids', (num_examples,), dtype=h5py.vlen_dtype(np.dtype('int8')), compression=hdf5_compression_method)
  masked_lm_positions = h5_writer.create_dataset('masked_lm_positions', (num_examples,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
  masked_lm_ids = h5_writer.create_dataset('masked_lm_ids', (num_examples,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
  next_sentence_labels = h5_writer.create_dataset('next_sentence_labels', data=np.zeros(num_examples, dtype="int8"), dtype='i1', compression=hdf5_compression_method)

  for i in tqdm(range(num_examples), total=num_examples):
    input_ids[i] = h5_ifile['input_ids'][i, :sum(h5_ifile['input_mask'][i])]
    segment_ids[i] = h5_ifile['segment_ids'][i, :sum(h5_ifile['input_mask'][i])]
    masked_lm_positions[i] = h5_ifile['masked_lm_positions'][i, :sum(h5_ifile['masked_lm_positions'][i]!=0)]
    masked_lm_ids[i] = h5_ifile['masked_lm_ids'][i, :sum(h5_ifile['masked_lm_positions'][i]!=0)]
    next_sentence_labels[i] = h5_ifile['next_sentence_labels'][i]

  h5_writer.flush()
  h5_writer.close()

  toc = time.time()
  logging.info("Converted {} examples in {:.2} sec".format(num_examples, toc - tic))
