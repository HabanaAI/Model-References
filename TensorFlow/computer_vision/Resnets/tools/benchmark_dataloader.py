# ******************************************************************************
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
# ******************************************************************************

import os
import ctypes
dynpatch_name = "dynpatch_prf_remote_call.so"
ctypes.CDLL(dynpatch_name, os.RTLD_LAZY + ctypes.RTLD_GLOBAL)

import logging
import tensorflow as tf
from absl import flags, app
from TensorFlow.common.utils import RangeTFProfilerHook
from TensorFlow.common.horovod_helpers import SynapseLoggerHook
from TensorFlow.computer_vision.Resnets.utils.logs import logger
import TensorFlow.computer_vision.Resnets.imagenet_main as imagenet_main
import TensorFlow.computer_vision.Resnets.utils.logs.hooks_helper as hooks_helper
from TensorFlow.common.horovod_helpers import hvd_init
from TensorFlow.common.multinode_helpers import comm_rank

log = logging.getLogger("horovod_benchmark")


def max_train_file_number(data_dir):
    """ find out how many imagenet tf records are available in data directory.
    For meaningful results of data loader tests it is obligatory to have at
    least some part of imagenet available locally."""

    min_file, max_file = 0, 1023
    imagenet_train_records = os.path.join(data_dir, "img_train")
    while min_file != max_file - 1:
        tfile = (max_file + min_file) // 2
        tfile_name = os.path.join(imagenet_train_records, "img_train-%05d-of-01024" % tfile)
        if os.path.isfile(tfile_name):
            log.info(f" found {tfile_name}")
            min_file = tfile
        else:
            log.info(f" not found {tfile_name}")
            max_file = tfile
    log.info(f"found {min_file} imagenet files under {imagenet_train_records}")
    return min_file

class MonkeypatchStub:
    @staticmethod
    def setitem(o, k, v):
        o[k] = v

    @staticmethod
    def setattr(o, k, v):
        setattr(o, k, v)

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[1], initializer="ones")
        super().build(input_shape)

    def call(self, x):
        return x * self.kernel

class ImagenetModelMock:
    def __init__(
        self,
        resnet_size,
        data_format=None,
        num_classes=imagenet_main.NUM_CLASSES,
        resnet_version=imagenet_main.resnet_model.DEFAULT_VERSION,
        dtype=imagenet_main.resnet_model.DEFAULT_DTYPE,
    ):
        log.info(f"ImagenetModelMock parameters {locals()}")
        self.num_classes = num_classes

    def __call__(self, inputs, training):
        prod = 1
        for d in inputs.shape[1:]:
            prod = prod * d
        flat = tf.reshape(inputs, (-1, prod))
        global_step =  tf.compat.v1.train.get_global_step()
        fg = tf.cast(global_step, tf.float32)
        offset = 20
        result = tf.slice(flat, (0, offset), (-1, self.num_classes))
        result = result / 127.0
        result = CustomLayer()(result) + fg
        result = tf.nn.relu(result)
        return result

def setup_hooks(enable_profiling, profiling_iter_cnt, profiling_warmup_steps=20):

    hooks = []

    if (enable_profiling):

        begin = profiling_warmup_steps + 1
        end = begin + profiling_iter_cnt

        synapse_logger_hook = SynapseLoggerHook(
            list(range(begin, end)), False)

        def get_synapse_logger_hook(**kwargs):
            return synapse_logger_hook
        MonkeypatchStub.setitem(hooks_helper.HOOKS,
                            "synapse_logger_hook", get_synapse_logger_hook)
        hooks.append("synapse_logger_hook")

        tf_profiler_hook = RangeTFProfilerHook(
            begin, profiling_iter_cnt, f"rank_{comm_rank()}")

        def get_tf_profiler_hook(**kwargs):
            return tf_profiler_hook
        MonkeypatchStub.setitem(hooks_helper.HOOKS,
                            "tf_profiler_hook", get_tf_profiler_hook)
        hooks.append("tf_profiler_hook")

    return ",".join(hooks)


def setup_env(allow_control_edges, bfloat16):
    if (bfloat16):
        os.environ["TF_ENABLE_BF16_CONVERSION"] = "1"
    if (allow_control_edges):
        os.environ["TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS"] = "1"
    os.environ["HBN_TF_REGISTER_DATASETOPS"] = "1"
    os.environ["LOCK_GAUDI_SYNAPSE_API"] = "1"

def benchmark_demo_data_loader(
    data_dir,
    batch_size=256,
    bfloat16=True,
    max_train_steps=1024,
    datasets_num_private_threads=False,
    allow_control_edges=True,
    enable_profiling=False,
    profiling_iter_cnt=20,
    profiling_warmup_steps=20,
    experimental_preloading=1
):
    hooks = setup_hooks(enable_profiling, profiling_iter_cnt,
                        profiling_warmup_steps=profiling_warmup_steps)

    setup_env(allow_control_edges, bfloat16)

    hvd_init()

    MonkeypatchStub.setattr(imagenet_main, "_NUM_TRAIN_FILES",
                        max_train_file_number(data_dir))

    MonkeypatchStub.setattr(imagenet_main, "ImagenetModel", ImagenetModelMock)

    tf.compat.v1.enable_resource_variables()

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    imagenet_main.define_imagenet_flags()
    flags.DEFINE_boolean("mini_imagenet", False, "mini ImageNet")

    argv = [
        "test",
        f"--model_dir=rank_{comm_rank()}",
        "--num_gpus=1",
        f"--data_dir={data_dir}",
        "--distribution_strategy=off",
        "--data_format=channels_last",
        f"--batch_size={batch_size}",
        "--return_before_eval=true",
        "--display_steps=100",
        "--use_horovod=true",
        f"--experimental_preloading={experimental_preloading}"
    ]

    if hooks:
        argv.append(f"--hooks={hooks}")

    if datasets_num_private_threads:
        argv.append(
            f"--datasets_num_private_threads={datasets_num_private_threads}")
    if max_train_steps:
        argv.append(f"--max_train_steps={max_train_steps}")

    flags.FLAGS(argv)

    imagenet_main.run_imagenet(flags.FLAGS)


if __name__ == "__main__":
    DATA_DIR_DEFAULT=os.path.expanduser("~/tensorflow_datasets/imagenet/tf_records/")
    def main():
        import argparse
        logging.basicConfig(level=logging.INFO)

        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--enable_profiling",
                            action='store_true', help="Run test with profiling hooks")
        parser.add_argument("-c", "--profiling_iter_cnt", default=20,
                            type=int, help="Number of iteration to profile")
        parser.add_argument("-w", "--profiling_warmup_steps", default=20,
                            type=int, help="Number of iteration before profiling starts")
        parser.add_argument("-b", "--batch_size", default=256,
                            type=int, help="Batch size.")
        parser.add_argument("-t", "--data_type",
                            choices=["fp32", "bf16"], type=str, default="bf16")
        parser.add_argument("-d", "--data_dir", type=str, default=DATA_DIR_DEFAULT,
                            help=f"Location of dataset (should point to tf_records). Default path: {DATA_DIR_DEFAULT}")
        parser.add_argument("-s", "--max_train_steps", type=int, default=None,
                            help="Max training steps.")
        parser.add_argument("-e", "--experimental_preloading", type=int, default=1,
                            help="Use experimental preloading.")

        args = parser.parse_args()

        benchmark_demo_data_loader(
            args.data_dir,
            batch_size=args.batch_size,
            bfloat16=(args.data_type == "bf16"),
            max_train_steps=args.max_train_steps,
            enable_profiling=args.enable_profiling,
            profiling_iter_cnt=args.profiling_iter_cnt,
            profiling_warmup_steps=args.profiling_warmup_steps,
            experimental_preloading=args.experimental_preloading)
    main()
