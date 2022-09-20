###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
import pytest
import tensorflow as tf
import habana_frameworks.tensorflow as htf

# Some common global test parameters
valueA = 1456.0
valueB = 32.0
expected_division_result = valueA / valueB


def format_tc(val):
    """Formats test case parametrization argument. Function intended to use with tc params, to
    print them clearly in pytest --collect-only. If function not used params in printed not by value
    but with appended integer. For example it would be dtype0, dtype1, etc."""
    if isinstance(val, tf.dtypes.DType):  # pylint: disable=no-member
        ret = repr(val)
        return ret.split(sep=".")[1]
    return str(val)


def setup_module():
    htf.load_habana_module()


@pytest.mark.parametrize("dtype", [tf.float32, tf.bfloat16], ids=format_tc)
def test_custom_div_op_function(custom_op_lib_path, dtype):
    # custom op lib should be loaded with load_op_library function from habana-tensorflow
    custom_op_lib = htf.load_op_library(custom_op_lib_path)
    custom_div = custom_op_lib.custom_div_op

    # Tensorflow function uses Graph optimization passes provided by both TF and Habana
    # One of the result of Habana optimizations is clustering of CustomDivOp
    # into an encapsulated Op called HabanaCluster
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=dtype), tf.TensorSpec(shape=None, dtype=dtype)])
    def run_function(a, b):
        return custom_div(a, b, example_attr=23)

    assert run_function(valueA, valueB).numpy() == expected_division_result


@pytest.mark.parametrize("dtype", [tf.float32, tf.bfloat16], ids=format_tc)
def test_custom_div_op_eager(custom_op_lib_path, dtype):
    # custom op lib should be loaded with load_op_library function from habana-tensorflow
    custom_op_lib = htf.load_op_library(custom_op_lib_path)
    custom_div = custom_op_lib.custom_div_op

    assert custom_div(tf.constant(valueA, dtype=dtype), tf.constant(valueB, dtype=dtype),
                      example_attr=42).numpy() == expected_division_result
