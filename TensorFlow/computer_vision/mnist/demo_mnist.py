import argparse
import os
import sys
import time

import tensorflow as tf
from data_provider import DataProvider
from mnist_model import MnistModel
import numpy as np

from demo.library_loader import load_habana_module

class BenchmarkData:
    def __init__(self):
        self.start_timer = 0
        self.stop_timer = 0
        self.iters_data = []

    def getIterationsNumber(self):
        return len(self.iters_data)

class MnistDemo:

    def __init__(self, args):
        log_info_devices = load_habana_module()
        print(f"Devices:\n {log_info_devices}")
        self.MnistDataProvider = DataProvider("mnist")
        self.data_set, self.info = self.MnistDataProvider.getData()
        self.model = MnistModel((None,) + self.info.features["image"].shape,
            self.info.features["label"].num_classes, tf.float32, args.optimizer)
        self.args = args
        self.num_steps = args.iterations
        self.display_step = 10
        self.batch_size = args.batch_size
        self.trainAndEval()

    def trainAndEval(self):

        if self.args.ops_placement:
            train_config = tf.compat.v1.ConfigProto(log_device_placement=True)
        else:
            train_config = tf.compat.v1.ConfigProto()

        benchmark = BenchmarkData()
        with tf.compat.v1.Session(config=train_config) as sess:
            self.trainModel(sess, benchmark)
            self.evaluateModel(sess, benchmark)
            sess.close()

    def trainModel(self, sess, benchmark):
        data_set_train_batches = self.data_set.train.batch(self.batch_size)
        iterator = tf.compat.v1.data.make_initializable_iterator(data_set_train_batches)
        next_batch = iterator.get_next()
        sess.run(self.model.init)
        sess.run(self.model.init_l)
        sess.run(iterator.initializer)
        epoch = 0
        step = 0
        print("Start training")
        benchmark.start_timer = time.time()
        training_chart = dict()
        while step < self.num_steps:
            try:
                data_batch = sess.run(next_batch)
                if len(data_batch["label"]) != self.batch_size:
                    print("End of dataset")
                    sess.run(iterator.initializer)
                    epoch = epoch + 1
                    continue
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                sess.run(iterator.initializer)
                epoch = epoch + 1
                continue
            batch_x = data_batch["image"]
            batch_y = data_batch["label"]

            iter_timer_start = time.time()
            sess.run(self.model.train_op, feed_dict={self.model.X: batch_x, self.model.labels: batch_y})
            iter_timer_dur = time.time() - iter_timer_start
            if step % self.display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                acc = sess.run([self.model.accuracy], feed_dict={self.model.X: batch_x, self.model.labels: batch_y})
                loss = sess.run([self.model.loss_op], feed_dict={self.model.X: batch_x, self.model.labels: batch_y})
                iter_acc = self.model.get_scalar_acc(acc)
                iter_time = time.time() - benchmark.start_timer
                benchmark.iters_data.append({'step': step,
                                             'loss': loss,
                                             'iter_acc': iter_acc,
                                             'iter_time': iter_time})
                print(
                    f"[{(iter_time):10.2f}sec.] Epoch {epoch}, Iteration {step:5}, " \
                    f"Minibatch Loss={loss[0]:16.6f}, Training Accuracy={iter_acc:10.6f}")
                if loss[0] is None:
                    training_chart[str(step)] = dict(
                        acc=float(iter_acc), duration=iter_timer_dur)
                else:
                    training_chart[str(step)] = dict(
                        acc=float(iter_acc), loss=float(loss[0]), duration=iter_timer_dur)
            step = step + 1
        benchmark.stop_timer = time.time()
        print("Stop training")

    def evaluateModel(self, sess, benchmark):
        # get batch for test
        data_set_test_batches = self.data_set.test.batch(self.batch_size)
        iterator = tf.compat.v1.data.make_initializable_iterator(data_set_test_batches)
        sess.run(iterator.initializer)
        next_batch = iterator.get_next()
        data_batch = sess.run(next_batch)
        batch_x = data_batch["image"]
        batch_y = data_batch["label"]
        acc = sess.run([self.model.accuracy], feed_dict={self.model.X: batch_x, self.model.labels: batch_y})
        acc = self.model.get_scalar_acc(acc)

def main():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.enable_resource_variables()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size [default=8]')
    parser.add_argument('--iterations', default=400, type=int,
                        help='Number of training iterations: [default=400]')
    parser.add_argument('--data_type', default='fp32',
                        help="Training data type: fp32, bf16 [default=fp32]")
    parser.add_argument('--ops_placement', default=False, type=bool,
                        help='Show ops placement [default=False]')
    parser.add_argument('--optimizer', default="Adam", type=str, choices=("Adam", "Momentum"),
                        help='Optimizer flavor. Selecting momenum also enables resource variables. [default=Adam]')

    args = parser.parse_args()

    if args.data_type == 'bf16':
        os.environ['TF_ENABLE_BF16_CONVERSION'] = '1'
    MnistDemo(args)

if __name__ == "__main__":
    main()
