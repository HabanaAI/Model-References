# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
###############################################################################
# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - script migration to Tensorflow 2.x version
# - included HPU horovod setup
# - added tensorboard logging functionality
# - added TumorCore, PeritumoralEdema and EnhancingTumor metrics to evaluation results
# - debug_train and debug_predict options have been removed

import os
import logging

import numpy as np
import tensorflow as tf

from dataset.data_loader import Dataset, CLASSES
from runtime.hooks import get_hooks
from runtime.arguments import parse_args
from runtime.setup import prepare_model_dir, build_estimator, setup_horovod, set_flags, get_logger


def parse_evaluation_results(result):
    data = {CLASSES[i]: result[CLASSES[i]] for i in range(len(CLASSES))}
    data['MeanDice'] = str(sum([result[CLASSES[i]] for i in range(len(CLASSES))]) / len(CLASSES))
    data['WholeTumor'] = str(result['WholeTumor'])
    data['TumorCore'] = str(data['TumorCore'])
    data['PeritumoralEdema'] = str(data['PeritumoralEdema'])
    data['EnhancingTumor'] = str(data['EnhancingTumor'])
    return data


def main():
    params = parse_args()
    tf.random.set_seed(params.seed)
    tf.get_logger().setLevel(logging.ERROR)

    params = setup_horovod(params)
    set_flags(params)
    model_dir = prepare_model_dir(params)
    logger = get_logger(params)

    dataset = Dataset(data_dir=params.data_dir,
                      batch_size=params.batch_size,
                      fold_idx=params.fold,
                      n_folds=params.num_folds,
                      params=params,
                      seed=params.seed)

    estimator = build_estimator(params, model_dir)

    if params.tensorboard_logging and (params.worker_id == 0 or params.log_all_workers):
        from TensorFlow.common.tb_utils import write_hparams_v1
        write_hparams_v1(params.log_dir, vars(params))

    if not params.benchmark:
        params.max_steps = params.max_steps // params.num_workers
    if 'train' in params.exec_mode:
        training_hooks = get_hooks(params, logger)

        estimator.train(
            input_fn=dataset.train_fn,
            steps=params.max_steps,
            hooks=training_hooks)

    if 'evaluate' in params.exec_mode:
        result = estimator.evaluate(input_fn=dataset.eval_fn, steps=dataset.eval_size)
        data = parse_evaluation_results(result)
        if params.worker_id == 0:
            logger.log(step=(), data=data)

    if 'predict' == params.exec_mode:
        inference_hooks = get_hooks(params, logger)
        if params.worker_id == 0:
            count = 1 if not params.benchmark else 2 * params.warmup_steps * params.batch_size // dataset.test_size
            predictions = estimator.predict(
                input_fn=lambda: dataset.test_fn(count=count,
                                                 drop_remainder=params.benchmark), hooks=inference_hooks)

            for idx, p in enumerate(predictions):
                volume = p['predictions']
                if not params.benchmark:
                    np.save(os.path.join(params.model_dir, "vol_{}.npy".format(idx)), volume)


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    main()
