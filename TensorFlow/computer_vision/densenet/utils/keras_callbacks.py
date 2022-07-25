###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

import tensorflow as tf
import numpy as np
import random as python_random
import pickle
import os
from tensorflow import keras
import json
from collections import ChainMap
import tensorflow.compat.v1.keras.backend as K
import time


def save_output_pickle(output: list, pickle_path: str, fetches: list) -> None:
    dump_list = []

    for idx, fetch in enumerate(fetches):
        dump_list.append((fetch, output[idx],))

    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    with open(pickle_path, "wb") as f_out:
        pickle.dump(dump_list, f_out)


class KerasTensorExtractionCallback(keras.callbacks.Callback):
    def __init__(self, model, json_config_file_path):
        wights_path = ''
        with open(json_config_file_path) as f:
            data = json.load(f)
            self.wights_path = data['weights']
            # Construction of graph inputs - <dictionary key: name, value: tensors>
            self.feed_dict = {}
            inputs_dict = dict(ChainMap(*data['inputs']))
            for input_name, path in inputs_dict.items():
                self.feed_dict[input_name] = pickle.load(open(path, 'rb'))
            self.dump_tensor_names = data['outputs']
            self.outputDumpPath = data['dump_path']
        self.model = model

    def get_input(self) -> np.ndarray:
        return self.feed_dict['input_1:0']

    def get_target(self)-> np.ndarray:
        return self.feed_dict['Logits_target:0']

    def on_train_batch_begin(self, batch, logs=None) -> None:
        # Force deterministic behavior
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        np.random.seed(0)
        python_random.seed(0)
        tf.random.set_seed(0)

        print('KerasTensorExtractionCallback: loading weights from: ' + self.wights_path)
        self.model.load_weights(self.wights_path)

        session = K.get_session()

        ops = session.graph.get_operations()

        print('Keras sbs callback: Running the session')
        output = session.run(self.dump_tensor_names, self.feed_dict)

        print('KerasTensorExtractionCallback: Saving tensors to - ' + self.outputDumpPath)
        save_output_pickle(output, self.outputDumpPath, self.dump_tensor_names)

        print("KerasTensorExtractionCallback: Exiting.")
        exit()

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))


class KerasMeasurePerfCallback(keras.callbacks.Callback):
    def __init__(self, model, batch):
        self.model = model
        self.batch = batch
        self.batch_start = 0
        self.batch_end = 0
        self.max_img_per_sec = 0
        self.average_img_per_sec = 0

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_train_batch_begin(self, batch, logs=None) -> None:
        keys = list(logs.keys())
        self.batch_start = start = time.time()
        print("Training: start of batch {}; Done comparing tensors. Exiting training.")

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        self.batch_end = time.time()
        batch_execution_time = self.batch_end - self.batch_start
        images_per_second = self.batch/batch_execution_time
        if images_per_second > self.max_img_per_sec:
            self.max_img_per_sec = images_per_second
        if batch > 0:
            self.average_img_per_sec = ((self.average_img_per_sec*(batch))+images_per_second)/(batch+1)
        else:
            self.average_img_per_sec = images_per_second
        print("Images per second         : {}".format(images_per_second))
        print("Max Images per second     : {}".format(self.max_img_per_sec))
        print("Average Images per second : {}".format(self.average_img_per_sec))
        print("    Batch execution time  : {}".format(batch_execution_time))
        print("    Batch size            : {}".format(self.batch))
