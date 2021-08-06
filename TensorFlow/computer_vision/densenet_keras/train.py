"""train.sh

This script is used to train the ImageNet models.
"""


import os
import time
import argparse
import sys
import tensorflow as tf

from config import config
from utils.utils import config_keras_backend_for_gpu, clear_keras_session
from utils.dataset import get_dataset
from utils.keras_callbacks import KerasMeasurePerfCallback, KerasTensorExtractionCallback
from models.models import get_batch_size
from models.models import get_iter_size
from models.models import get_lr_func
from models.models import get_initial_lr
from models.models import get_final_lr
from models.models import get_weight_decay
from models.models import get_optimizer
from models.models import get_training_model

from TensorFlow.common.library_loader import load_habana_module
from TensorFlow.common.tb_utils import (
    TensorBoardWithHParamsV1, ExamplesPerSecondKerasHook)

from tqdm import tqdm

DESCRIPTION = """For example:
$ python3 train.py --dataset_dir  ${HOME}/data/ILSVRC2012/tfrecords \
                   --dropout_rate 0.4 \
                   --optimizer    adam \
                   --epsilon      1e-1 \
                   --label_smoothing \
                   --batch_size   32 \
                   --iter_size    1 \
                   --lr_sched     exp \
                   --initial_lr   1e-2 \
                   --final_lr     1e-5 \
                   --weight_decay 2e-4 \
                   --epochs       60
"""



class ValCB(tf.keras.callbacks.Callback):
    def __init__(self, total_val_steps):
        super(ValCB).__init__()
        self.total_val_steps = total_val_steps


def train(model_name, dropout_rate, optim_name, epsilon,
          label_smoothing, use_lookahead, batch_size, iter_size,
          lr_sched, initial_lr, final_lr,
          weight_decay, epochs, iterations, dataset_dir, skip_eval, eval_checkpoint,
          run_on_hpu, measure_perf, extract_tensors_cfg_file_path, bfloat16,
          train_subset, val_subset, save_summary_steps):

    if not run_on_hpu:
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    """Prepare data and train the model."""
    batch_size   = get_batch_size(model_name, batch_size)
    iter_size    = get_iter_size(model_name, iter_size)
    initial_lr   = get_initial_lr(model_name, initial_lr)
    final_lr     = get_final_lr(model_name, final_lr)
    optimizer    = get_optimizer(model_name, optim_name, initial_lr, epsilon)
    weight_decay = get_weight_decay(model_name, weight_decay)

    # get training and validation data
    ds_train = get_dataset(dataset_dir, train_subset, batch_size)
    if skip_eval:
        ds_valid = None
    else:
        ds_valid = get_dataset(dataset_dir, val_subset, batch_size)

    # instantiate training callbacks
    lrate = get_lr_func(epochs, lr_sched, initial_lr, final_lr)
    save_name = model_name if not model_name.endswith('.h5') else \
        os.path.split(model_name)[-1].split('.')[0].split('-')[0]
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config.SAVE_DIR, save_name) + '-ckpt-{epoch:03d}.h5',
        monitor='train_loss')

    if iterations:
        steps_per_epoch = iterations
        print(f"Changing steps per epoch to {steps_per_epoch}")
    else:
        steps_per_epoch = 1281167 // batch_size

    if skip_eval:
        val_steps = 0
    else:
        val_steps = 50000 // batch_size

    # build model and do training
    get_training_model_kwargs = {
        "model_name": model_name,
        "dropout_rate": dropout_rate,
        "optimizer": optimizer,
        "label_smoothing": label_smoothing,
        "use_lookahead": use_lookahead,
        "iter_size": iter_size,
        "weight_decay": weight_decay,
        "batch_size": batch_size
    }

    if not run_on_hpu:
        with strategy.scope():
            model = get_training_model(**get_training_model_kwargs)
    else:
        if bfloat16:
            # Bf16 conversion, full list
            os.environ['TF_ENABLE_BF16_CONVERSION'] = 'full'
        else:
            os.environ['TF_ENABLE_BF16_CONVERSION'] = "false"

        print("train: Set TF_ENABLE_BF16_CONVERSION: " + os.environ.get('TF_ENABLE_BF16_CONVERSION'))
        model = get_training_model(**get_training_model_kwargs)

    if eval_checkpoint != None:
        model.load_weights(eval_checkpoint)
        results = model.evaluate(x=ds_valid, steps=val_steps)
        print("Test loss, Test acc:", results)
        exit()

    x = ds_train
    y = None
    callbacks = [lrate, model_ckpt]
    shuffle = True
    if measure_perf:
        callbacks += [KerasMeasurePerfCallback(model, batch_size)]

    if extract_tensors_cfg_file_path != None:
        tenorsExtractionCallback = KerasTensorExtractionCallback(model, extract_tensors_cfg_file_path)
        callbacks += [tenorsExtractionCallback]
        x = tenorsExtractionCallback.get_input()
        y = tenorsExtractionCallback.get_target()
        steps_per_epoch = 1
        epochs = 1
        ds_valid = None
        val_steps = 0
        shuffle = False

    if save_summary_steps is not None and save_summary_steps > 0:
        callbacks += [
            TensorBoardWithHParamsV1(
                get_training_model_kwargs, log_dir=config.LOG_DIR,
                update_freq=save_summary_steps, profile_batch=0),
            ExamplesPerSecondKerasHook(
                save_summary_steps, output_dir=config.LOG_DIR,
                batch_size=batch_size),
        ]

    model.fit(
        x=x,
        y=y,
        steps_per_epoch = steps_per_epoch,
        validation_data = ds_valid,
        validation_steps = val_steps,
        callbacks=callbacks,
        epochs=epochs,
        shuffle=shuffle)

    # training finished
    model.save('{}/{}-model-final.h5'.format(config.SAVE_DIR, save_name))


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset_dir', type=str,
                        default=config.DEFAULT_DATASET_DIR)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'rmsprop'])
    parser.add_argument('--epsilon', type=float, default=1e-1)
    parser.add_argument('--label_smoothing', action='store_true')
    parser.add_argument('--use_lookahead', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--iter_size', type=int, default=1)
    parser.add_argument('--lr_sched', type=str, default='steps',
                        choices=['linear', 'exp', 'steps'])
    parser.add_argument('--initial_lr', type=float, default=5e-2)
    parser.add_argument('--final_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=90,
                        help='total number of epochs for training [1]')
    parser.add_argument('--model', type=str, default='densenet121')
    parser.add_argument('--run_on_hpu', type=str, default='True')
    parser.add_argument('--bfloat16', type=str, default='True')
    parser.add_argument('--log_device_placement', action='store_true')
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--measure_perf', action='store_true')
    parser.add_argument('--extract_tensors', help="--extract_tensors <Path to dump extracted tensors>.",
                        type=str)
    parser.add_argument('--only_eval', help="--only_eval <Path to checkpoint>. Performs model evaluation only.",
                        type=str)
    parser.add_argument('--iterations', help="Sets number of iterations per epoch",
                        type=int)
    parser.add_argument('--train_subset', type=str, default='train')
    parser.add_argument('--val_subset', type=str, default='validation')
    parser.add_argument('--save_summary_steps', type=int, default=None,
                        help='Steps between saving summaries to TensorBoard. '
                             'When None, logging to TensorBoard is disabled. '
                             'Enabling this option might affect the performance.')
    args = parser.parse_args()

    args.bfloat16 = eval(args.bfloat16)
    args.run_on_hpu = eval(args.run_on_hpu)

    if args.skip_eval or args.only_eval == None:
        tf.keras.backend.set_learning_phase(True)

    if args.run_on_hpu:
        log_info_devices = load_habana_module()
        print(f"Devices:\n {log_info_devices}")
    else:
        config_keras_backend_for_gpu()
    tf.debugging.set_log_device_placement(args.log_device_placement)

    if args.use_lookahead and args.iter_size > 1:
        raise ValueError('cannot set both use_lookahead and iter_size')

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print("model:           " + str(args.model))
    print("dropout_rate:    " + str(args.dropout_rate))
    print("optimizer:       " + str(args.optimizer))
    print("epsilon:         " + str(args.epsilon))
    print("label_smoothing: " + str(args.label_smoothing))
    print("use_lookahead:   " + str(args.use_lookahead))
    print("batch_size:      " + str(args.batch_size))
    print("iter_size:       " + str(args.iter_size))
    print("lr_sched:        " + str(args.lr_sched))
    print("initial_lr:      " + str(args.initial_lr))
    print("final_lr:        " + str(args.final_lr))
    print("weight_decay:    " + str(args.weight_decay))
    print("epochs:          " + str(args.epochs))
    print("iterations:      " + str(args.iterations))
    print("dataset_dir:     " + str(args.dataset_dir))
    print("skip_eval:       " + str(args.skip_eval))
    print("only_eval:       " + str(args.only_eval))
    print("run_on_hpu:      " + str(args.run_on_hpu))
    print("bfloat16:        " + str(args.bfloat16))
    print("train subset:    " + str(args.train_subset))
    print("val subset:      " + str(args.val_subset))

    train(args.model, args.dropout_rate, args.optimizer, args.epsilon,
          args.label_smoothing, args.use_lookahead,
          args.batch_size, args.iter_size,
          args.lr_sched, args.initial_lr, args.final_lr,
          args.weight_decay, args.epochs, args.iterations, args.dataset_dir,
          args.skip_eval, args.only_eval, args.run_on_hpu,
          args.measure_perf, args.extract_tensors, args.bfloat16,
          args.train_subset, args.val_subset, args.save_summary_steps)
    clear_keras_session()


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.enable_resource_variables()
    main()
