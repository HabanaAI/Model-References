# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company

from packaging import version
import tensorflow as tf

if version.parse(tf.__version__) <= version.parse("2.12.1"):
    from tensorflow.python.framework.ops import convert_to_tensor_v2
else:
    from tensorflow.python.framework.tensor_conversion import convert_to_tensor_v2
