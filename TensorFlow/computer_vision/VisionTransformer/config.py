###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

from types import SimpleNamespace
config = SimpleNamespace()

# Subdirectory name for saving trained weights and models.
config.SAVE_DIR = 'saves'

# Subdirectory name for saving TensorBoard log files.
config.LOG_DIR = 'logs'

# Default path to the ImageNet TFRecords dataset files.
config.DEFAULT_DATASET_DIR = '/data/tensorflow/imagenet/tf_records'

# Path to weights cache directory. ~/.keras is used if None.
config.WEIGHTS_DIR = None

# Number of parallel workers for generating training/validation data.
config.NUM_DATA_WORKERS = 128

# Do image data augmentation or not.
config.DATA_AUGMENTATION = True

# Enable deterministic behavior.
config.DETERMINISTIC = False

# Seed to be used by random functions.
config.SEED = None
