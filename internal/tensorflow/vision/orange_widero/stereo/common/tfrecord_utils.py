import json
import tensorflow as tf
from stereo.common.s3_utils import my_open


def _bytes_feature(value):
    """returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def example_np_to_example(example_np, ds_format):
    """
    convert example_np to tf example (string)
    :param example_np: dictionry
    :param ds_format: dict of the dataset format
    :return:
    """
    feature = {k: _bytes_feature(example_np[k].astype(ds_format[k]['dtype']).tobytes())
               for k in ds_format.keys()}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def parse_example(example, ds_format, to_numpy=False):
    """
    Parses a single tf.Example
    """
    read_features = {k: tf.io.FixedLenFeature([], tf.string) for k in ds_format.keys()}
    parsed = tf.io.parse_single_example(serialized=example, features=read_features)
    for k in parsed:
        if ds_format[k]['dtype'].startswith('string'):
            if int(tf.version.VERSION.split('.')[1]) > 12:
                parsed[k] = tf.strings.unicode_encode(tf.cast(tf.reshape(
                    tf.io.decode_raw(parsed[k], out_type=tf.uint8), (-1,)), tf.int32), output_encoding='UTF-8')
        else:
            parsed[k] = tf.io.decode_raw(parsed[k], out_type=ds_format[k]['dtype'])
            parsed[k] = tf.reshape(parsed[k], ds_format[k]['shape'])
        if to_numpy:
            parsed[k] = parsed[k].numpy()

    return parsed


def write_tfrecord(examples, tfrecord_path, ds_format, compress=True, verbose=False):
    """
    write TFRecord file from npz file list
    :param examples: list of np dicts or tf examples (serialized strings)
    :param tfrecord_path: TFRecord filename
    :param ds_format: either path (or s3 url) to ds_format json file or loaded json dict
    :param compress: true for writing compressed TFrecords
    :param verbose:
    """
    if isinstance(ds_format, str):
        with my_open(ds_format, 'rb') as fp:
            ds_format = json.load(fp)
    if compress:
        comp_type = tf.compat.v1.io.TFRecordCompressionType.GZIP
    else:
        comp_type = tf.compat.v1.io.TFRecordCompressionType.NONE
    with tf.io.TFRecordWriter(tfrecord_path, options=comp_type) as writer:
        for example in examples:
            example = example.numpy() if isinstance(example, tf.Tensor) else example_np_to_example(example, ds_format)
            writer.write(example)
    if verbose:
        print('Wrote {n} examples to tfrecord {f}'.format(**{'n': len(examples), 'f': tfrecord_path}))


def read_tfrecord(tfrecord_path, ds_format, parse=False, to_numpy=False, compression_type="GZIP", verbose=False):
    """
    Read all examples contained in tfrecord_path
    :param tfrecord_path: path (or s3 url) to tfrecord file
    :param ds_format: either path (or s3 url) to ds_format json file or loaded json dict
    :param parse: When True, run parse_example() on each entry of the tf_record
    :param to_numpy: When True, and assuming parse is also True, returned examples are dicts of numpy arrays. Assumes
        tf.enable_eager_execution() was called before running.
    :param compression_type: this is tf.data.TFRecordDataset()'s compression_type (default: "GZIP")
    :param verbose: (default: False)
    """
    if isinstance(ds_format, str):
        with my_open(ds_format, 'rb') as fp:
            ds_format = json.load(fp)

    if to_numpy:  # make sure we're in eager execution mode
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.compat.v1.enable_eager_execution(config=session_conf)

    tfrecord = tf.data.TFRecordDataset(tfrecord_path, compression_type=compression_type)
    iterator = tf.compat.v1.data.make_one_shot_iterator(tfrecord)
    examples = list(iterator) if not parse else \
        [parse_example(e, ds_format, to_numpy=to_numpy) for e in list(iterator)]

    if verbose:
        print('Read {n} examples from tfrecord {f}'.format(**{'n': len(examples), 'f': tfrecord_path}))

    return examples
