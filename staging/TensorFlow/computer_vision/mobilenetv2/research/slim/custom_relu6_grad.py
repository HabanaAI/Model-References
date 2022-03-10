import tensorflow as tf
from tensorflow.python.framework import ops
from habana_frameworks.tensorflow import habana_ops
custom_op_lib = tf.load_op_library("../../../../../../TensorFlow/examples/custom_op/build/lib/libhpu_custom_relu6_op.so")
custom_relu6_grad = custom_op_lib.custom_relu6_grad_op

@ops.RegisterGradient("CustomRelu6Op")
def _CustomRelu6Grad(op, grad):
  return custom_relu6_grad(grad, op.outputs[0])
