# Copyright (c) 2021 Habana Labs Ltd., an Intel Company

import os, sys
import math
import json
from keras_segmentation.data_utils.data_loader import image_segmentation_generator, \
     verify_segmentation_dataset, DATA_LOADER_SEED, create_segmentation_list, cached_image_generator
import glob
import six
import numpy as np
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
import tensorflow.keras
tensorflow.keras.backend.set_learning_phase(1)
import tensorflow as tf
import time
import pickle
from keras_segmentation.f1 import FBetaScore
from tensorflow.keras.optimizers import Adam

Distributed = False
PERF_EPOCH_SKIP = 1

class EpochMonitor(Callback):
    def __init__(self, num_steps, bs, rank, rank_size):
        super(EpochMonitor, self).__init__()
        self.num_steps = num_steps
        self.bs = bs
        self.rank = rank
        self.val_step = 0
        self.total_ips = np.zeros(rank_size)
        self.max_epoch = np.zeros(rank_size)
        self.tm = np.zeros(rank_size)
        self.epoch = 0
        self.start_epoch_time = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        print("Starting epoch: ", epoch, " on rank ", self.rank)
        self.start_epoch_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        tm = time.time() - self.start_epoch_time
        tpb = tm/self.num_steps
        tpi = tpb / self.bs
        ips = 1/tpi
        if epoch >= PERF_EPOCH_SKIP:
            self.total_ips[self.rank] += ips
            self.tm[self.rank] += tm
        self.max_epoch[self.rank] = epoch
        print("\nTime per batch(s):", tpb, " on rank ", self.rank)
        print("Time per image(s):", tpi, " on rank ", self.rank)
        print("Images per second:", ips, " on rank ", self.rank)
        print("Epoch time(s):", tm, " on rank ", self.rank, " on steps ", self.num_steps)
    def on_train_end(self, logs=None):
        epochs = self.max_epoch[self.rank] + 1 - PERF_EPOCH_SKIP
        OVERALL_AVG_EPS = self.total_ips[self.rank]/epochs
        OVERALL_AVG_TPB = self.tm[self.rank]/(self.num_steps * epochs)
        OVERALL_AVG_TPI = OVERALL_AVG_TPB/self.bs
        print('Average time per batch in seconds:', OVERALL_AVG_TPB, " on rank ", self.rank, " with skip ",PERF_EPOCH_SKIP," epoch")
        print('Average time per image in seconds:', OVERALL_AVG_TPI, " on rank ", self.rank, " with skip ",PERF_EPOCH_SKIP," epoch")
        print('Average images per sec:', OVERALL_AVG_EPS, " on rank ", self.rank, " with skip ",PERF_EPOCH_SKIP," epoch")

tf.random.set_seed(DATA_LOADER_SEED)

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    all_checkpoint_files = [ ff.replace(".index" , "" ) for ff in all_checkpoint_files ] # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


def masked_categorical_crossentropy(gt, pr):
    from tf.keras.losses import categorical_crossentropy
    mask = 1 - gt[:, :, 0]
    return categorical_crossentropy(gt, pr) * mask

class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))

def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=None,
          val_steps_per_epoch=None,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adam',
          do_augment=False,
          augmentation_name="aug_all",
          data_type='fp32',
          tb_location=None,
          deterministic=False,
          model_dir=None,
          dump_config=None,
          distributed=False,
          use_upsampling=False,
          loss_type=0,
          train_engine='hpu',
          not_cached=False,
          learning_rate=0.001):

    if train_engine == 'hpu':
        from habana_frameworks.tensorflow import load_habana_module
        load_habana_module()
        print("Loaded HPU modules")
        from TensorFlow.common.debug import dump_callback
        # For Habana Model runner hooks
        from TensorFlow.common.tb_utils import (TensorBoardWithHParamsV2, ExamplesPerSecondKerasHookV2)
    else:
        if data_type == 'amp':
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("Enabling AMP mixed precision float16...")
        class dump_callback(object):
            def __init__(self, file_name):
                pass
            def __enter__(self):
                pass
            def __exit__(self, type, value, traceback):
                pass


    if data_type=='bf16' and train_engine == 'hpu':
        bf16_json = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../bf16_segnet.json')
        os.environ ['TF_BF16_CONVERSION'] = os.environ.get('TF_BF16_CONVERSION', bf16_json)
        print("Setting BF16:",os.getenv('TF_BF16_CONVERSION'))

    shard_id = 0
    num_shards = 1

    if distributed:
        import horovod.tensorflow.keras as hvd
        print("hvd init")
        hvd.init()
        if train_engine == 'gpu':
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
            print("Set memory growth for GPUS")

        shard_id = hvd.rank()
        num_shards =hvd.size()
        if num_shards == 1:
            print("Distributed training requested but horovod init not success")
            exit()

    print("num_shards: " + str(num_shards) + " shard_id: " + str(shard_id))

    from keras_segmentation.models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width, batch_size=batch_size, use_upsampling=use_upsampling, loss_type=loss_type)
        else:
            model = model_from_name[model](n_classes, batch_size=batch_size, use_upsampling=use_upsampling, loss_type=loss_type)

    #model.save('my_segnet_model.h5')
    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if steps_per_epoch is None:
        steps_per_epoch  =len(os.listdir(train_images))//(batch_size*num_shards)
    if val_steps_per_epoch is None:
        val_steps_per_epoch = len(os.listdir(val_images))//(batch_size*num_shards)

    print("Steps per epoch: " + str(steps_per_epoch))

    def optimized_xent_loss_custom_grad(ytrue, ypred):
        @tf.custom_gradient
        def loss_without_mean(ytrue, ypred):
            with tf.name_scope("softmax_cross_entropy"):
                logits_t = tf.transpose(ypred, perm=(0, 1, 3, 2), name="logits_t") # BS H N W
                reduce_max = tf.reduce_max(logits_t, 2, name="reduce_max") # BS H W
                max_logits = tf.expand_dims(reduce_max, 3) # BS H W 1
                shifted_logits = tf.subtract(ypred, max_logits, name="shifted_logits") # BS H W N
                exp_shifted_logits = tf.math.exp(
                    shifted_logits, name="exp_shifted_logits") # BS H W N
                reduce_sum_filter = tf.fill([1,1,n_classes, 1], 1.0)
                reduce_sum_filter = tf.cast(reduce_sum_filter, tf.float16) if data_type == 'amp' else reduce_sum_filter
                sum_exp = tf.nn.conv2d(
                    exp_shifted_logits, reduce_sum_filter, strides=1, padding="VALID", name="sum_exp") # BS H W 1
                log_sum_exp = tf.math.log(sum_exp, name="log_sum_exp") # BS H W 1
                shifted_logits2 = tf.nn.conv2d(shifted_logits * ytrue, reduce_sum_filter, strides=1, padding="VALID", name="shifted_logits2") # BS H W 1
                loss = tf.subtract(log_sum_exp, shifted_logits2, name="loss/sub") # BS H W 1

                def custom_grad(dy): # dy is BS H W 1
                    with tf.name_scope("gradients/softmax_cross_entropy"):
                        div = tf.math.truediv(exp_shifted_logits, sum_exp, name="div") # BS H W N
                        sub = tf.math.subtract(div, ytrue, name="sub")  # BS H W N
                        ret = tf.math.multiply(sub, dy, name="mul")
                    return -dy*shifted_logits, ret
                return loss, custom_grad

        return tf.math.reduce_mean(loss_without_mean(ytrue, ypred))

    if validate:
        assert val_images is not None
        assert val_annotations is not None


    if optimizer_name is not None:

        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        elif loss_type == 1:
            loss_k = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        elif loss_type == 2:
            loss_k = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        else:
            loss_k = optimized_xent_loss_custom_grad


        print(optimizer_name)
        print("Learning rate ", learning_rate)
        if num_shards > 1:
            optimizer = Adam(lr= learning_rate)
            optimizer_name = hvd.DistributedOptimizer(optimizer)

        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        status = model.load_weights(load_weights)
        print(status)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes, deterministic)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes, deterministic)
            assert verified

    if not_cached:
        train_gen = image_segmentation_generator(
            train_images, train_annotations,  batch_size,  n_classes,
            input_height, input_width, output_height, output_width, deterministic,
            do_augment=do_augment, augmentation_name=augmentation_name,
            num_shards=num_shards, shard_id=shard_id, loss_type=loss_type)
    else:
        train_gen = image_segmentation_generator(
            train_images, train_annotations,  1,  n_classes,
            input_height, input_width, output_height, output_width, deterministic,
            do_augment=do_augment, augmentation_name=augmentation_name,
            num_shards=num_shards, shard_id=shard_id, loss_type=loss_type)

        train_gen = cached_image_generator(train_gen, num_shards, shard_id, batch_size, len(os.listdir(train_images)), deterministic)

    callbacks = []

    if num_shards > 1:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
        callbacks.append(hvd.callbacks.MetricAverageCallback())

    callbacks.append(CheckpointsCallback(checkpoints_path))
    if steps_per_epoch > 1:
        callbacks.append(EpochMonitor(num_steps=steps_per_epoch, bs=batch_size, rank=shard_id, rank_size=num_shards))
    #if shard_id == 0:
    #    callbacks.append(ModelCheckpoint( self.checkpoints_path, monitor='loss', verbose=2, mode='min', save_best_only=True, save_weights_only=True))
    log_path = None
    if model_dir is not None:
        hparams = {
                "model_name": model,
                "optimizer": optimizer_name,
                "batch_size": batch_size
            }

        if train_engine == 'hpu':
            if distributed:
                print("TensorBoardWithHParamsV2:",model_dir,":",hvd.rank())
                log_path = model_dir+"/worker_"+str(hvd.rank())
            else:
                log_path = model_dir+"/worker_0"
            print("log_path",log_path)
            callbacks += [
                TensorBoardWithHParamsV2(
                    hparams, log_dir=log_path,
                    update_freq=5),
                ExamplesPerSecondKerasHookV2(
                    5, batch_size=batch_size, output_dir=log_path)
            ]

    if tb_location != '':
        tensorboard_callback = TensorBoard(log_dir=tb_location, histogram_freq=1)
        callbacks.append(tensorboard_callback)
        print("TB:" , tb_location)

    start_training = time.time()
    if not validate:
        with dump_callback(dump_config):
            history=model.fit(train_gen, steps_per_epoch = steps_per_epoch,
                                epochs=epochs, callbacks=callbacks, verbose=1 if shard_id==0 else 0)
        with open('./trainHistoryDict_'+str(shard_id), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        if loss_type == 1:
            print('Eval for LOSS_FUNC_TYPE=1 is WIP')
            exit()

        #Validation
        if not_cached:
            val_gen = image_segmentation_generator(
                val_images, val_annotations,  batch_size,
                n_classes, input_height, input_width, output_height, output_width, deterministic,
                num_shards=num_shard, shard_id=shard_id, loss_type=loss_type)
        else:
            val_gen = image_segmentation_generator(
                val_images, val_annotations,  1,
                n_classes, input_height, input_width, output_height, output_width, deterministic,
                num_shards=num_shards, shard_id=shard_id, loss_type=loss_type)
            val_gen = cached_image_generator(val_gen, 1, 0, batch_size, len(os.listdir(val_images)))

        f1_metric = FBetaScore(num_classes=n_classes)
        model.compile(loss=model.loss, metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None), f1_metric])
        test_loss, test_acc, test_f1 = model.evaluate(val_gen, steps=(len(os.listdir(val_images))//batch_size))
        train_loss, train_acc, train_f1 = model.evaluate(train_gen, steps=(len(os.listdir(train_images))//batch_size))
        print(f'test loss : {test_loss}, test accuracy : {test_acc}, test f1 : {test_f1}')
        print(f'train loss : {train_loss}, train accuracy : {train_acc}, train f1 : {train_f1}')
        # Add test accuracy to tf events for QA/Kibana reporting
        if log_path is not None:
            test_summary_writer = tf.summary.create_file_writer(log_path)
            with test_summary_writer.as_default():
                tf.summary.scalar('eval_acc', test_acc, step=1)
    else:
        if not_cached:
            val_gen = image_segmentation_generator(
                val_images, val_annotations,  batch_size,
                n_classes, input_height, input_width, output_height, output_width, deterministic,
                num_shards=num_shards, shard_id=shard_id, loss_type=loss_type)
        else:
            val_gen = image_segmentation_generator(
                val_images, val_annotations,  1,
                n_classes, input_height, input_width, output_height, output_width, deterministic,
                num_shards=num_shards, shard_id=shard_id, loss_type=loss_type)
            val_gen = cached_image_generator(val_gen, num_shards, shard_id, batch_size, len(os.listdir(val_images)), deterministic)

        model.fit(train_gen,
                            steps_per_epoch = steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=callbacks,
                            use_multiprocessing=gen_use_multiprocessing, verbose=1 if shard_id==0 else 0)

    stop_training = time.time()
    print('Training Time in seconds:', (stop_training-start_training))
