---
title: Resnet changes
---

Originally, scripts were taken from [Tensorflow
Github](https://github.com/tensorflow/models.git), tag v1.13.0

Files used:

-   imagenet\_main.py
-   imagenet\_preprocessing.py
-   resnet\_model.py
-   resnet\_run\_loop.py

All of above files were converted to TF2 by using
[tf\_upgrade\_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool.
Additionally, some other changes were committed for specific files.

Main changes are:

1.  Added HPU support
2.  Added Horovod support for multinode
3.  Added MLPerf support
4.  Added Resnext, see resnext.html for details

imagenet\_main.py
=================

1.  Added

    ``` {.sourceCode .python}
    import sys
    ```

    For sys.path.append

2.  Changed

    ``` {.sourceCode .python}
    import tensorflow as tf  # pylint: disable=g-bad-import-order
    ```

    to

    ``` {.sourceCode .python}
    import tensorflow as tf
    ```

3.  Changed

    ``` {.sourceCode .python}
    from official.utils.flags import core as flags_core
    from official.utils.logs import logger
    from official.resnet import imagenet_preprocessing
    from official.resnet import resnet_model
    from official.resnet import resnet_run_loop
    ```

    to

    ``` {.sourceCode .python}
    from TensorFlow.computer_vision.Resnets import imagenet_preprocessing
    from TensorFlow.computer_vision.Resnets import resnet_model
    from TensorFlow.computer_vision.Resnets import resnet_run_loop
    from TensorFlow.computer_vision.Resnets.utils.flags import core as flags_core
    from TensorFlow.computer_vision.Resnets.utils.logs import logger
    ```

    Changed local path

4.  Added

    ``` {.sourceCode .python}
    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(curr_path, '..'))
    ```

5.  Added

    ``` {.sourceCode .python}
    from TensorFlow.common.horovod_helpers import hvd, hvd_init, horovod_enabled, synapse_logger_init
    ```

    For Horovod support for multinode

6.  Added

    ``` {.sourceCode .python}
    from TensorFlow.common.utils import disable_session_recovery
    ```

7.  Added

    ``` {.sourceCode .python}
    from TensorFlow.common.library_loader import load_habana_module  # noqa
    ```

    For Habana HPU support

8.  Changed

    ``` {.sourceCode .python}
    if is_training:
      return [
          os.path.join(data_dir, 'train-%05d-of-01024' % i)
          for i in range(_NUM_TRAIN_FILES)]
    else:
      return [
          os.path.join(data_dir, 'validation-%05d-of-00128' % i)
          for i in range(128)]
    ```

    to

    ``` {.sourceCode .python}
    if flags.FLAGS.mini_imagenet:
      if is_training:
        return [
            os.path.join(data_dir, 'imagenet_train_subset/imagenet_train_subset-%05d-of-01024' % i)
            for i in range(_NUM_TRAIN_FILES)]
      else:
        return [
            os.path.join(data_dir, 'imagenet_val_subset/imagenet_val_subset-%05d-of-00128' % i)
            for i in range(128)]
    else:
      if is_training:
        return [
            os.path.join(data_dir, 'img_train/img_train-%05d-of-01024' % i)
            for i in range(_NUM_TRAIN_FILES)]
      else:
        return [
            os.path.join(data_dir, 'img_val/img_val-%05d-of-00128' % i)
            for i in range(128)]
    ```

    For mini\_imagenet support. Also path to dataset was changed.

9.  Changed

    ``` {.sourceCode .python}
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                        default_value=''),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                            default_value=-1),
    'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
    ```

    to

    ``` {.sourceCode .python}
    'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
    'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                               default_value=-1),
    'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                              default_value=''),
    ```

10. Changed

    ``` {.sourceCode .python}
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    ```

    to

    ``` {.sourceCode .python}
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    ```

11. Changed

    ``` {.sourceCode .python}
    features = tf.parse_single_example(example_serialized, feature_map)
    ```

    to

    ``` {.sourceCode .python}
    features = tf.io.parse_single_example(serialized=example_serialized, features=feature_map)
    ```

12. Changed

    ``` {.sourceCode .python}
    bbox = tf.transpose(bbox, [0, 2, 1])
    ```

    to

    ``` {.sourceCode .python}
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])
    ```

13. Changed

    ``` {.sourceCode .python}
    def input_fn(is_training, data_dir, batch_size, num_epochs=1,
                 dtype=tf.float32, datasets_num_private_threads=None,
                 num_parallel_batches=1, parse_record_fn=parse_record):
    ```

    to

    ``` {.sourceCode .python}
    def input_fn(is_training,
                 data_dir,
                 batch_size,
                 num_epochs=1,
                 dtype=tf.float32,
                 datasets_num_private_threads=None,
                 parse_record_fn=parse_record,
                 input_context=None,
                 drop_remainder=False,
                 tf_data_experimental_slack=False,
                 experimental_preloading=False,
                 dataset_fn=None):
    ```

    5 parameters added at the end. num\_parallel\_batches parameter
    removed.

14. Removed

    ``` {.sourceCode .python}
    num_parallel_batches: Number of parallel batches for tf.data.
    ```

    From input\_fn() parameters description

15. Added

    ``` {.sourceCode .python}
    input_context: A `tf.distribute.InputContext` object passed in by
      `tf.distribute.Strategy`.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.
    ```

    To input\_fn() parameters description

16. Changed

    ``` {.sourceCode .python}
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    ```

    to

    ``` {.sourceCode .python}
    if dataset_fn is None:
      filenames = get_filenames(is_training, data_dir)
      dataset = tf.data.Dataset.from_tensor_slices(filenames)
    else:
      dataset = dataset_fn()

    if is_training and horovod_enabled():
      dataset = dataset.shard(hvd.size(), hvd.rank())

    if input_context:
      tf.compat.v1.logging.info(
          'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
              input_context.input_pipeline_id, input_context.num_input_pipelines))
      dataset = dataset.shard(input_context.num_input_pipelines,
                              input_context.input_pipeline_id)
    ```

17. Changed

    ``` {.sourceCode .python}
    # cycle_length = 10 means 10 files will be read and deserialized in parallel.
    # This number is low enough to not cause too much contention on small systems
    # but high enough to provide the benefits of parallelization. You may want
    # to increase this number if you have a large number of CPU cores.
    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10))
    ```

    to

    ``` {.sourceCode .python}
    # cycle_length = 10 means that up to 10 files will be read and deserialized in
    # parallel. You may want to increase this number if you have a large number of
    # CPU cores.
    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=10,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ```

18. Changed

    ``` {.sourceCode .python}
    return resnet_run_loop.process_record_dataset(
        (...)
        num_parallel_batches=num_parallel_batches
    ```

    to

    ``` {.sourceCode .python}
    return resnet_run_loop.process_record_dataset(
        (...)
        drop_remainder=drop_remainder,
        tf_data_experimental_slack=tf_data_experimental_slack,
        experimental_preloading=experimental_preloading
    ```

    Added 3 new parameters. Parameter num\_parallel\_batches is removed.

19. Added

    ``` {.sourceCode .python}
    model_type=resnet_model.DEFAULT_MODEL_TYPE,
    ```

    Added parameter to ImagenetModel::\_\_init\_\_. This is for
    selecting between resnet and resnext.

20. Added

    ``` {.sourceCode .python}
    model_type=model_type,
    ```

    To call to super(ImagenetModel, self).\_\_init\_\_()

21. Changed

    ``` {.sourceCode .python}
    if params['fine_tune']:
    ```

    to

    ``` {.sourceCode .python}
    if params['fine_tune'] or ('disable_warmup' in params and params['disable_warmup']):
    ```

22. Added

    ``` {.sourceCode .python}
    # According to https://arxiv.org/abs/1706.02677 and our internal experiments,
    # the best accuracy results for more than 16 devices are achieved when base_lr == 0.1
    if horovod_enabled() and hvd.size() > 16:
      base_lr = .1

    # Used for ResNeXt101-32x4d
    if params['use_cosine_lr']:
      base_lr = .256

    if horovod_enabled():
      total_batch_size = params['batch_size'] * hvd.size()
    else:
      total_batch_size = params['batch_size'] * params.get('num_workers', 1)
    ```

    For horovod and resnext support

23. Changed

    ``` {.sourceCode .python}
    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=256,
        num_images=NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4], warmup=warmup, base_lr=base_lr)
    ```

    to

    ``` {.sourceCode .python}
    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=total_batch_size,
        batch_denom=256, num_images=NUM_IMAGES['train'],
        boundary_epochs=[30, 60, 80, 90],
        train_epochs=params['train_epochs'],
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
        warmup=warmup,
        warmup_epochs=params['warmup_epochs'],
        base_lr=base_lr,
        use_cosine_lr=params['use_cosine_lr'])
    ```

24. Changed

    ``` {.sourceCode .python}
    weight_decay=1e-4,
    ```

    to

    ``` {.sourceCode .python}
    weight_decay=flags.FLAGS.weight_decay,
    ```

25. Changed

    ``` {.sourceCode .python}
    momentum=0.9,
    ```

    to

    ``` {.sourceCode .python}
    momentum=flags.FLAGS.momentum,
    ```

26. Added

    ``` {.sourceCode .python}
    model_type=params['model_type'],
    ```

    In call to resnet\_run\_loop.resnet\_model\_fn()

27. Changed

    ``` {.sourceCode .python}
    fine_tune=params['fine_tune']
    ```

    to

    ``` {.sourceCode .python}
    fine_tune=params['fine_tune'],
    label_smoothing=flags.FLAGS.label_smoothing
    ```

    In call to resnet\_run\_loop.resnet\_model\_fn()

28. Changed

    ``` {.sourceCode .python}
    resnet_run_loop.define_resnet_flags(
        resnet_size_choices=['18', '34', '50', '101', '152', '200'])
    ```

    to

    ``` {.sourceCode .python}
    resnet_run_loop.define_resnet_flags(
        resnet_size_choices=['18', '34', '50', '101', '152', '200'],
        dynamic_loss_scale=True,
        fp16_implementation=False)
    ```

29. Added

    ``` {.sourceCode .python}
    flags.DEFINE_bool(
        name='use_horovod', short_name='hvd', default=False,
        help=flags_core.help_wrap(
            'Run training using horovod.'))
    ```

    Flag for enabling horovod support

30. Changed

    ``` {.sourceCode .python}
    flags_core.set_defaults(train_epochs=90)
    ```

    to

    ``` {.sourceCode .python}
    flags_core.set_defaults(train_epochs=10)  # 90
    ```

31. Added

    ``` {.sourceCode .python}
    Returns:
      Dict of results of the run.  Contains the keys `eval_results` and
        `train_hooks`. `eval_results` contains accuracy (top_1) and
        accuracy_top_5. `train_hooks` is a list the instances of hooks used during
        training.
    ```

    Added comment to run\_imagenet()

32. Added

    ``` {.sourceCode .python}
    if flags.FLAGS.is_mlperf_enabled:
      tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    else:
      tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    ```

    For mlperf support

33. Added

    ``` {.sourceCode .python}
    if flags_obj.use_horovod:
      assert flags_obj.no_hpu == False, "Horovod without HPU is not supported in helpers."
      hvd_init()
    else:
      synapse_logger_init()
    ```

    For horovod support

34. Added

    ``` {.sourceCode .python}
    if not flags_obj.no_hpu:
      log_info_devices = load_habana_module()
      print(f"Devices:\n {log_info_devices}")
    ```

    For Habana HPU support

35. Changed

    ``` {.sourceCode .python}
    resnet_run_loop.resnet_main(
    ```

    to

    ``` {.sourceCode .python}
    result = resnet_run_loop.resnet_main(
    (...)
    return result
    ```

36. Changed

    ``` {.sourceCode .python}
    with logger.benchmark_context(flags.FLAGS):
    ```

    to

    ``` {.sourceCode .python}
    with logger.benchmark_context(flags.FLAGS), disable_session_recovery():
    ```

    In main()

37. Changed

    ``` {.sourceCode .python}
    tf.logging.set_verbosity(tf.logging.INFO)
    ```

    to

    ``` {.sourceCode .python}
    tf.compat.v1.disable_eager_execution()
    # Opt-in to TF 2.0 feature - improved variables with a well defined memory model
    tf.compat.v1.enable_resource_variables()

    flags.DEFINE_boolean('mini_imagenet', False, 'mini ImageNet')

    print("Script arguments:", " ".join(sys.argv[1:]))
    ```

    In main code section

imagenet\_preprocessing.py
==========================

1.  Changed

    ``` {.sourceCode .python}
    def _decode_crop_and_flip(image_buffer, bbox, num_channels):
    ```

    to

    ``` {.sourceCode .python}
    def _decode_crop_and_flip(image_buffer, bbox, num_channels, seed=None):
    ```

2.  Changed

    ``` {.sourceCode .python}
    tf.image.extract_jpeg_shape(image_buffer),
    ```

    to

    ``` {.sourceCode .python}
    image_size=tf.image.extract_jpeg_shape(image_buffer),
    ```

3.  Changed

    ``` {.sourceCode .python}
    use_image_if_no_bounding_boxes=True)
    ```

    to

    ``` {.sourceCode .python}
    use_image_if_no_bounding_boxes=True,
    seed=0 if seed is None else seed)
    ```

4.  Changed

    ``` {.sourceCode .python}
    shape = tf.shape(image)
    ```

    to

    ``` {.sourceCode .python}
    shape = tf.shape(input=image)
    ```

5.  Changed

    ``` {.sourceCode .python}
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)
    ```

    to

    ``` {.sourceCode .python}
    # Note(b/130245863): we explicitly call `broadcast` instead of simply
    # expanding dimensions for better performance.
    means = tf.broadcast_to(means, tf.shape(input=image))
    ```

6.  Changed

    ``` {.sourceCode .python}
    shape = tf.shape(image)
    ```

    to

    ``` {.sourceCode .python}
    shape = tf.shape(input=image)
    ```

7.  Changed

    ``` {.sourceCode .python}
    return tf.image.resize_images(
    ```

    to

    ``` {.sourceCode .python}
    return tf.compat.v1.image.resize(
    ```

8.  Changed

    ``` {.sourceCode .python}
    num_channels, is_training=False):
    ```

    to

    ``` {.sourceCode .python}
    num_channels, is_training=False, training_seed=None):
    ```

9.  Changed

    ``` {.sourceCode .python}
    image = _decode_crop_and_flip(image_buffer, bbox, num_channels)
    ```

    to

    ``` {.sourceCode .python}
    image = _decode_crop_and_flip(image_buffer, bbox, num_channels, training_seed)
    ```

resnet\_model.py
================

1.  Added

    ``` {.sourceCode .python}
    DEFAULT_MODEL_TYPE = 'resnet'
    ```

2.  Changed

    ``` {.sourceCode .python}
    CASTABLE_TYPES = (tf.float16,)
    ```

    to

    ``` {.sourceCode .python}
    CASTABLE_TYPES = (tf.float16, tf.bfloat16)
    ```

3.  Changed

    ``` {.sourceCode .python}
    return tf.layers.batch_normalization(
    ```

    to

    ``` {.sourceCode .python}
    return tf.compat.v1.layers.batch_normalization(
    ```

4.  Changed

    ``` {.sourceCode .python}
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
    ```

    to

    ``` {.sourceCode .python}
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
    ```

5.  Changed

    ``` {.sourceCode .python}
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    ```

    to

    ``` {.sourceCode .python}
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
    ```

6.  Changed

    ``` {.sourceCode .python}
    if strides > 1:
      inputs = fixed_padding(inputs, kernel_size, data_format)
    ```

    to

    ``` {.sourceCode .python}
    #if strides > 1:
    #  inputs = fixed_padding(inputs, kernel_size, data_format)
    ```

7.  Changed

    ``` {.sourceCode .python}
    return tf.layers.conv2d(
    ```

    to

    ``` {.sourceCode .python}
    return tf.compat.v1.layers.conv2d(
    ```

8.  Changed

    ``` {.sourceCode .python}
    padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
    kernel_initializer=tf.variance_scaling_initializer(),
    ```

    to

    ``` {.sourceCode .python}
    padding=('SAME' if strides == 1 else 'SAME'), use_bias=False,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
    ```

9.  Added

    ``` {.sourceCode .python}
    def conv2d_groups_fixed_padding(inputs, filters, kernel_size, strides, data_format, split=1):
      """Strided 2-D convolution with groups and explicit padding."""
      # The padding is consistent and is based only on `kernel_size`, not on the
      # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
      #if strides > 1:
      #  inputs = fixed_padding(inputs, kernel_size, data_format)

      if split == 1:
        return conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format)

      else:
        padding = 'SAME' # !! Check

        in_shape = inputs.get_shape().as_list()
        channel_axis = 1 if data_format == 'channels_first' else 3
        in_channel = in_shape[channel_axis]

        assert in_channel % split == 0, in_channel

        out_channel = filters
        assert out_channel % split == 0, out_channel

        filter_shape = [kernel_size, kernel_size, in_channel // split, out_channel]
        if data_format == 'channels_first':
          stride = [1, 1, strides, strides]
        else:
          stride = [1, strides, strides, 1]

        kernel_initializer = tf.compat.v1.variance_scaling_initializer() # !! Check
        inputs_dtype = inputs.dtype

        W = tf.Variable(kernel_initializer(shape=filter_shape, dtype=inputs_dtype), trainable=True)

        return tf.nn.conv2d(inputs, W, stride, padding)
    ```

10. Added

    ``` {.sourceCode .python}
    def _resnext_bottleneck_block(inputs, filters, training, projection_shortcut,
                             strides, data_format):
      shortcut = inputs

      if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                              data_format=data_format)

      #  l = Conv2D('conv1', l, ch_out * 2, 1, strides=1, activation=BNReLU)
      inputs = conv2d_groups_fixed_padding(
          inputs=inputs, filters=filters*2, kernel_size=1, strides=1,
          data_format=data_format)

      inputs = batch_norm(inputs, training, data_format)
      inputs = tf.nn.relu(inputs)

      # l = Conv2D('conv2', l, ch_out * 2, 3, strides=stride, activation=BNReLU, split=32)
      inputs = conv2d_groups_fixed_padding(
          inputs=inputs, filters=filters*2, kernel_size=3, strides=strides,
          data_format=data_format, split=32)

      inputs = batch_norm(inputs, training, data_format)
      inputs = tf.nn.relu(inputs)

      # l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))
      inputs = conv2d_groups_fixed_padding(
          inputs=inputs, filters=filters*4, kernel_size=1, strides=1,
          data_format=data_format)

      # out = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))
      inputs = batch_norm(inputs, training, data_format)
      inputs += shortcut
      inputs = tf.nn.relu(inputs)

      return inputs
    ```

11. Changed

    ``` {.sourceCode .python}
    resnet_version=DEFAULT_VERSION, data_format=None,
    ```

    to

    ``` {.sourceCode .python}
    resnet_version=DEFAULT_VERSION,
    model_type=DEFAULT_MODEL_TYPE,
    data_format=None,
    ```

12. Changed

    ``` {.sourceCode .python}
    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1:
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = _bottleneck_block_v2
    else:
      if resnet_version == 1:
        self.block_fn = _building_block_v1
    ```

    to

    ``` {.sourceCode .python}
    if model_type not in ('resnet', 'resnext'):
      raise ValueError(
          'Only resnet and resnext are supported.')

    if model_type == 'resnext':
      self.bottleneck = True
      self.block_fn = _resnext_bottleneck_block

    elif model_type == 'resnet':
      self.bottleneck = bottleneck

      if bottleneck:
        if resnet_version == 1:
          self.block_fn = _bottleneck_block_v1
        else:
          self.block_fn = _bottleneck_block_v2
    ```

13. Changed

    ``` {.sourceCode .python}
    self.block_fn = _building_block_v2
    ```

    to

    ``` {.sourceCode .python}
    if resnet_version == 1:
      self.block_fn = _building_block_v1
    else:
      self.block_fn = _building_block_v2
    ```

14. Changed

    ``` {.sourceCode .python}
    return tf.variable_scope('resnet_model',
                             custom_getter=self._custom_dtype_getter)
    ```

    to

    ``` {.sourceCode .python}
    return tf.compat.v1.variable_scope('resnet_model',
                                       custom_getter=self._custom_dtype_getter)
    ```

15. Changed

    ``` {.sourceCode .python}
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    ```

    to

    ``` {.sourceCode .python}
    inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])
    ```

16. Changed

    ``` {.sourceCode .python}
    inputs = tf.layers.max_pooling2d(
    ```

    to

    ``` {.sourceCode .python}
    inputs = tf.compat.v1.layers.max_pooling2d(
    ```

17. Changed

    ``` {.sourceCode .python}
    inputs = tf.reduce_mean(inputs, axes, keepdims=True)
    ```

    to

    ``` {.sourceCode .python}
    inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
    ```

18. Changed

    ``` {.sourceCode .python}
    inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
    ```

    to

    ``` {.sourceCode .python}
    inputs = tf.compat.v1.layers.dense(inputs=inputs, units=self.num_classes)
    ```

resnet\_run\_loop.py
====================

1.  Removed

    ``` {.sourceCode .python}
    # pylint: disable=g-bad-import-order
    ```

2.  Removed

    ``` {.sourceCode .python}
    from tensorflow.contrib.data.python.ops import threadpool
    ```

3.  Changed

    ``` {.sourceCode .python}
    from official.resnet import resnet_model
    from official.utils.flags import core as flags_core
    from official.utils.export import export
    from official.utils.logs import hooks_helper
    from official.utils.logs import logger
    from official.resnet import imagenet_preprocessing
    from official.utils.misc import distribution_utils
    from official.utils.misc import model_helpers
    ```

    to

    ``` {.sourceCode .python}
    from TensorFlow.computer_vision.Resnets import imagenet_preprocessing
    from TensorFlow.computer_vision.Resnets import resnet_model
    from TensorFlow.computer_vision.Resnets.utils import export
    from TensorFlow.computer_vision.Resnets.utils.flags import core as flags_core
    from TensorFlow.computer_vision.Resnets.utils.logs import hooks_helper
    from TensorFlow.computer_vision.Resnets.utils.logs import logger
    from TensorFlow.computer_vision.Resnets.utils.misc import distribution_utils
    from TensorFlow.computer_vision.Resnets.utils.misc import model_helpers
    from TensorFlow.common.horovod_helpers import hvd, horovod_enabled
    from TensorFlow.common.habana_estimator import HabanaEstimator
    from TensorFlow.common.str2bool import condition_env_var
    from TensorFlow.computer_vision.Resnets import imagenet_main
    import TensorFlow.computer_vision.Resnets.utils.optimizers.LARSOptimizer as lars

    if os.environ.get('USE_MLPERF', 0) == '1':
        from mlperf_compliance import mlperf_log
        from mlperf_logging import mllog
        from mlperf_compliance import tf_mlperf_log

        str_hvd_rank = str(hvd.rank()) if horovod_enabled() else "0"
        mllogger = mllog.get_mllogger()
        filenames = "resnet50v1.5.log-" + str_hvd_rank
        mllog.config(filename=filenames)
        workername = "worker" + str_hvd_rank
        mllog.config(
            default_namespace = workername,
            default_stack_offset = 1,
            default_clear_line = False,
            root_dir = os.path.normpath(
               os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")))

    _NUM_EXAMPLES_NAME = "num_examples"
    ```

4.  Changed

    ``` {.sourceCode .python}
    num_parallel_batches=1):
    ```

    to

    ``` {.sourceCode .python}
    drop_remainder=False,
    tf_data_experimental_slack=False,
    experimental_preloading=False):
    ```

5.  Changed

    ``` {.sourceCode .python}
    num_parallel_batches: Number of parallel batches for tf.data.
    ```

    to

    ``` {.sourceCode .python}
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.
    tf_data_experimental_slack: Whether to enable tf.data's
      `experimental_slack` option.
    ```

6.  Added

    ``` {.sourceCode .python}
    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
      options = tf.data.Options()
      options.experimental_threading.private_threadpool_size = (
          datasets_num_private_threads)
      dataset = dataset.with_options(options)
      tf.compat.v1.logging.info('datasets_num_private_threads: %s',
                                datasets_num_private_threads)


    if not experimental_preloading:
      # Disable intra-op parallelism to optimize for throughput instead of latency.
      options = tf.data.Options()
      options.experimental_threading.max_intra_op_parallelism = 1
      dataset = dataset.with_options(options)
      # Prefetches a batch at a time to smooth out the time taken to load input
      # files for shuffling and processing.
      dataset = dataset.prefetch(buffer_size=batch_size)
    ```

7.  Removed

    ``` {.sourceCode .python}
    # Prefetches a batch at a time to smooth out the time taken to load input
    # files for shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    ```

8.  Added

    ``` {.sourceCode .python}
    if horovod_enabled():
      # Repeats the dataset. Due to sharding in multinode, training is related
      # directly to the number of max iterations not to number of epochs.
      dataset = dataset.repeat()
    else:
      # Repeats the dataset for the number of epochs to train.
      dataset = dataset.repeat(num_epochs)
    ```

9.  Changed

    ``` {.sourceCode .python}
    # Repeats the dataset for the number of epochs to train.
    dataset = dataset.repeat(num_epochs)
    ```

    to

    ``` {.sourceCode .python}
    num_parallel_calls = 16 if horovod_enabled() else tf.data.experimental.AUTOTUNE
    ```

10. Changed

    ``` {.sourceCode .python}
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False))
    ```

    to

    ``` {.sourceCode .python}
    dataset = dataset.map(
        lambda value: parse_record_fn(value, is_training, dtype),
        num_parallel_calls=num_parallel_calls, deterministic=False)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    ```

11. Removed

    ``` {.sourceCode .python}
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    ```

12. Changed

    ``` {.sourceCode .python}
    # Defines a specific size thread pool for tf.data operations.
    if datasets_num_private_threads:
      tf.logging.info('datasets_num_private_threads: %s',
                      datasets_num_private_threads)
      dataset = threadpool.override_threadpool(
          dataset,
          threadpool.PrivateThreadPool(
              datasets_num_private_threads,
              display_name='input_pipeline_thread_pool'))
    ```

    to

    ``` {.sourceCode .python}
    if experimental_preloading:
      device = "/device:HPU:0"
      dataset = dataset.apply(tf.data.experimental.prefetch_to_device(device))
    else:
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      if tf_data_experimental_slack:
        options = tf.data.Options()
        options.experimental_slack = True
        dataset = dataset.with_options(options)
    ```

13. Changed

    ``` {.sourceCode .python}
    inputs = tf.truncated_normal(
    ```

    to

    ``` {.sourceCode .python}
    inputs = tf.random.truncated_normal(
    ```

14. Changed

    ``` {.sourceCode .python}
    labels = tf.random_uniform(
    ```

    to

    ``` {.sourceCode .python}
    labels = tf.random.uniform(
    ```

15. Changed

    ``` {.sourceCode .python}
    data = data.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    ```

    to

    ``` {.sourceCode .python}
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ```

16. Changed

    ``` {.sourceCode .python}
    image_bytes_list = tf.placeholder(
    ```

    to

    ``` {.sourceCode .python}
    image_bytes_list = tf.compat.v1.placeholder(
    ```

17. Changed

    ``` {.sourceCode .python}
    tf.logging.info('Logical CPU cores: %s', cpu_count)
    ```

    to

    ``` {.sourceCode .python}
    tf.compat.v1.logging.info('Logical CPU cores: %s', cpu_count)
    ```

18. Changed

    ``` {.sourceCode .python}
    tf.logging.info('TF_GPU_THREAD_COUNT: %s', os.environ['TF_GPU_THREAD_COUNT'])
    tf.logging.info('TF_GPU_THREAD_MODE: %s', os.environ['TF_GPU_THREAD_MODE'])
    ```

    to

    ``` {.sourceCode .python}
    tf.compat.v1.logging.info('TF_GPU_THREAD_COUNT: %s',
                              os.environ['TF_GPU_THREAD_COUNT'])
    tf.compat.v1.logging.info('TF_GPU_THREAD_MODE: %s',
                              os.environ['TF_GPU_THREAD_MODE'])
    ```

19. Changed

    ``` {.sourceCode .python}
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates,
    base_lr=0.1, warmup=False):
    ```

    to

    ``` {.sourceCode .python}
    batch_size, batch_denom, num_images, boundary_epochs, train_epochs,
    decay_rates, base_lr=0.1, warmup=False, warmup_epochs=5, use_cosine_lr=False):
    ```

20. Changed

    ``` {.sourceCode .python}
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    ```

    to

    ``` {.sourceCode .python}
    if use_cosine_lr:
      tf.compat.v1.logging.info("Using cosine decay learning rate schedule")
      decay_steps = int(batches_per_epoch * train_epochs)
      lr = tf.compat.v1.train.cosine_decay(
        initial_learning_rate, global_step, decay_steps)
    else:
      if not flags.FLAGS.is_mlperf_enabled:
        tf.compat.v1.logging.info("Using piecewise learning rate schedule")
      lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, vals)
    ```

21. Changed

    ``` {.sourceCode .python}
    warmup_steps = int(batches_per_epoch * 5)
    ```

    to

    ``` {.sourceCode .python}
    if not flags.FLAGS.is_mlperf_enabled:
        tf.compat.v1.logging.info("Using warmup for %d epochs", warmup_epochs)
    warmup_steps = int(batches_per_epoch * warmup_epochs)
    ```

22. Changed

    ``` {.sourceCode .python}
    return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)
    ```

    to

    ``` {.sourceCode .python}
    return tf.cond(pred=global_step < warmup_steps,
                   true_fn=lambda: warmup_lr,
                   false_fn=lambda: lr)
    ```

23. Added

    ``` {.sourceCode .python}
    def poly_rate_fn(global_step):
      """Handles linear scaling rule, gradual warmup, and LR decay.

      The learning rate starts at 0, then it increases linearly per step.  After
      FLAGS.poly_warmup_epochs, we reach the base learning rate (scaled to account
      for batch size). The learning rate is then decayed using a polynomial rate
      decay schedule with power 2.0.

      Args:
        global_step: the current global_step

      Returns:
        returns the current learning rate
      """

      # Learning rate schedule for LARS polynomial schedule
      if flags.FLAGS.batch_size < 8192:
        plr = flags.FLAGS.start_learning_rate
        w_epochs = flags.FLAGS.warmup_epochs
      elif flags.FLAGS.batch_size < 16384:
        plr = 10.0
        w_epochs = 5
      elif flags.FLAGS.batch_size < 32768:
        plr = 25.0
        w_epochs = 5
      else:
        plr = 32.0
        w_epochs = 14

      w_steps = int(w_epochs * batches_per_epoch)
      wrate = (plr * tf.cast(global_step, tf.float32) / tf.cast(
          w_steps, tf.float32))

      # TODO(pkanwar): use a flag to help calc num_epochs.
      num_epochs = flags.FLAGS.epochs_for_lars
      train_steps = batches_per_epoch * num_epochs

      min_step = tf.constant(1, dtype=tf.int64)
      decay_steps = tf.maximum(min_step, tf.subtract(global_step, w_steps))
      poly_rate = tf.compat.v1.train.polynomial_decay(
          plr,
          decay_steps,
          train_steps - w_steps + 1,
          power=2.0)

      if flags.FLAGS.is_mlperf_enabled:
        mllogger.event(key=mllog.constants.OPT_BASE_LR, value=plr)
        mllogger.event(key=mllog.constants.LARS_OPT_LR_DECAY_POLY_POWER, value=2.0)
        mllogger.event(key=mllog.constants.LARS_OPT_END_LR, value=0.0001)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_EPOCHS, value=w_epochs)

      return tf.compat.v1.where(global_step <= w_steps, wrate, poly_rate)

    # For LARS we have a new learning rate schedule
    if flags.FLAGS.enable_lars:
      return poly_rate_fn
    ```

24. Changed

    ``` {.sourceCode .python}
    loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
    fine_tune=False):
    ```

    to

    ``` {.sourceCode .python}
    loss_filter_fn=None, model_type=resnet_model.DEFAULT_MODEL_TYPE,
    dtype=resnet_model.DEFAULT_DTYPE,
    fine_tune=False, label_smoothing=0.0):
    ```

25. Added

    ``` {.sourceCode .python}
    label_smoothing: If greater than 0 then smooth the labels.
    ```

26. Added

    ``` {.sourceCode .python}
    # Uncomment the following lines if you want to write images to summary,
    # we turned it off for performance reason
    ```

27. Changed

    ``` {.sourceCode .python}
    tf.summary.image('images', features, max_outputs=6)
    # Checks that features/images have same data type being used for calculations.
    assert features.dtype == dtype
    ```

    to

    ``` {.sourceCode .python}
    # tf.compat.v1.summary.image('images',
    #     (features, tf.cast(features, tf.float32)) [features.dtype == tf.bfloat16],
    #     max_outputs=6)

    if features.dtype != tf.bfloat16:
      # Checks that features/images have same data type being used for calculations.
      assert features.dtype == dtype
    ```

28. Changed

    ``` {.sourceCode .python}
    dtype=dtype)
    ```

    to

    ``` {.sourceCode .python}
    model_type=model_type, dtype=dtype)
    ```

29. Added

    ``` {.sourceCode .python}
    if flags.FLAGS.is_mlperf_enabled:
      num_examples_metric = tf_mlperf_log.sum_metric(tensor=tf.shape(input=logits)[0], name=_NUM_EXAMPLES_NAME)
    ```

30. Changed

    ``` {.sourceCode .python}
    'classes': tf.argmax(logits, axis=1),
    ```

    to

    ``` {.sourceCode .python}
    'classes': tf.argmax(input=logits, axis=1),
    ```

31. Changed

    ``` {.sourceCode .python}
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    ```

    to

    ``` {.sourceCode .python}
    labels = tf.cast(labels, tf.int32)

    if label_smoothing != 0.0:
      one_hot_labels = tf.one_hot(labels, 1001)
      cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=one_hot_labels,
          label_smoothing=label_smoothing)
    else:
      cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
          logits=logits, labels=labels)
    ```

32. Changed

    ``` {.sourceCode .python}
    tf.summary.scalar('cross_entropy', cross_entropy)
    ```

    to

    ``` {.sourceCode .python}
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)
    ```

33. Changed

    ``` {.sourceCode .python}
    [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
     if loss_filter_fn(v.name)])
    ```

    > tf.summary.scalar('l2\_loss', l2\_loss)

    to

    ``` {.sourceCode .python}
    [
        tf.nn.l2_loss(tf.cast(v, tf.float32))
        for v in tf.compat.v1.trainable_variables()
        if loss_filter_fn(v.name)
    ])
    ```

    > tf.compat.v1.summary.scalar('l2\_loss', l2\_loss)

34. Changed

    ``` {.sourceCode .python}
    global_step = tf.train.get_or_create_global_step()
    ```

    to

    ``` {.sourceCode .python}
    global_step = tf.compat.v1.train.get_or_create_global_step()
    ```

35. Changed

    ``` {.sourceCode .python}
    tf.summary.scalar('learning_rate', learning_rate)
    ```

    to

    ``` {.sourceCode .python}
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)

    if flags.FLAGS.enable_lars:
      tf.compat.v1.logging.info('Using LARS Optimizer.')
      optimizer = lars.LARSOptimizer(
          learning_rate,
          momentum=momentum,
          weight_decay=weight_decay,
          skip_list=['batch_normalization', 'bias'])

      if flags.FLAGS.is_mlperf_enabled:
        mllogger.event(key=mllog.constants.OPT_NAME, value=mllog.constants.LARS)
        mllogger.event(key=mllog.constants.LARS_EPSILON, value=0.0)
        mllogger.event(key=mllog.constants.LARS_OPT_WEIGHT_DECAY, value=weight_decay)
    else:
      optimizer = tf.compat.v1.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=momentum
      )

    fp16_implementation = getattr(flags.FLAGS, 'fp16_implementation', None)
    if fp16_implementation == 'graph_rewrite':
      optimizer = (
          tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(
              optimizer, loss_scale=loss_scale))
    ```

36. Changed

    ``` {.sourceCode .python}
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
      optimizer = hvd.DistributedOptimizer(optimizer)
    ```

37. Changed

    ``` {.sourceCode .python}
    if loss_scale != 1:
    ```

    to

    ``` {.sourceCode .python}
    if loss_scale != 1 and fp16_implementation != 'graph_rewrite':
    ```

38. Changed

    ``` {.sourceCode .python}
    grad_vars = optimizer.compute_gradients(loss)
    ```

    to

    ``` {.sourceCode .python}
    grad_vars = optimizer.compute_gradients(loss*loss_scale)
    ```

39. Added

    ``` {.sourceCode .python}
    grad_vars = [(grad / loss_scale, var) for grad, var in grad_vars]
    ```

40. Changed

    ``` {.sourceCode .python}
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
    ```

    to

    ``` {.sourceCode .python}
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    if flags.FLAGS.is_mlperf_enabled:
      train_op = tf.group(minimize_op, update_ops, num_examples_metric[1])
    else:
      train_op = tf.group(minimize_op, update_ops)
    ```

41. Changed

    ``` {.sourceCode .python}
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                    targets=labels,
                                                    k=5,
                                                    name='top_5_op'))
    ```

    to

    ``` {.sourceCode .python}
    accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.compat.v1.metrics.mean(
        tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))
    ```

42. Added

    ``` {.sourceCode .python}
    if flags.FLAGS.is_mlperf_enabled:
      metrics.update({_NUM_EXAMPLES_NAME: num_examples_metric})
    ```

43. Changed

    ``` {.sourceCode .python}
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])
    ```

    to

    ``` {.sourceCode .python}
    tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
    tf.compat.v1.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])
    ```

44. Changed

    ``` {.sourceCode .python}
    Dict of results of the run.
    ```

    to

    ``` {.sourceCode .python}
    Dict of results of the run.  Contains the keys `eval_results` and
    train\_hooks. eval\_results contains accuracy (top\_1) and
    accuracy\_top\_5. train\_hooks is a list the instances of hooks
    used during training.
    ```

45. Added

    ``` {.sourceCode .python}
    experimental_preloading = flags_obj.experimental_preloading
    ```

46. Added

    ``` {.sourceCode .python}
    # Configures cluster spec for distribution strategy.
    num_workers = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                                       flags_obj.task_index)
    ```

47. Changed

    ``` {.sourceCode .python}
    session_config = tf.ConfigProto(
    ```

    to

    ``` {.sourceCode .python}
    session_config = tf.compat.v1.ConfigProto(
    ```

48. Changed

    ``` {.sourceCode .python}
    allow_soft_placement=True)
    ```

    to

    ``` {.sourceCode .python}
    allow_soft_placement=not experimental_preloading)
      if horovod_enabled():
        # The Scoped Allocator Optimization is enabled by default unless disabled by a flag.
        if not condition_env_var('TF_DISABLE_SCOPED_ALLOCATOR', default=False):
          from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=import-error
          session_config.graph_options.rewrite_options.scoped_allocator_optimization = rewriter_config_pb2.RewriterConfig.ON
          enable_op = session_config.graph_options.rewrite_options.scoped_allocator_opts.enable_op
          del enable_op[:]
          enable_op.append("HorovodAllreduce")
    ```

49. Changed

    ``` {.sourceCode .python}
    flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)
    ```

    to

    ``` {.sourceCode .python}
    distribution_strategy=flags_obj.distribution_strategy,
    num_gpus=flags_core.get_num_gpus(flags_obj),
    num_workers=num_workers,
    all_reduce_alg=flags_obj.all_reduce_alg,
    num_packs=flags_obj.num_packs)
    ```

50. Changed

    ``` {.sourceCode .python}
    save_checkpoints_secs=60*60*24)
    ```

    to

    ``` {.sourceCode .python}
    log_step_count_steps=flags_obj.display_steps,
    save_checkpoints_secs=None,
    save_checkpoints_steps=flags_obj.save_checkpoint_steps)
    ```

51. Changed

    ``` {.sourceCode .python}
    if flags_obj.pretrained_model_checkpoint_path is not None:
      warm_start_settings = tf.estimator.WarmStartSettings(
          flags_obj.pretrained_model_checkpoint_path,
          vars_to_warm_start='^(?!.*dense)')
    ```

    to

    ``` {.sourceCode .python}
    # if flags_obj.pretrained_model_checkpoint_path is not None:
    #   warm_start_settings = tf.estimator.WarmStartSettings(
    #       flags_obj.pretrained_model_checkpoint_path,
    #       vars_to_warm_start='^(?!.*dense)')
    # else:
    #   warm_start_settings = None
    warm_start_settings = None

    model_dir=flags_obj.model_dir

    if horovod_enabled():
      model_dir="{}/rank_{}".format(flags_obj.model_dir, hvd.rank())

    if experimental_preloading:
      SelectedEstimator = HabanaEstimator
    ```

52. Changed

    ``` {.sourceCode .python}
    warm_start_settings = None
    ```

    to

    ``` {.sourceCode .python}
    SelectedEstimator = tf.estimator.Estimator
    ```

53. Changed

    ``` {.sourceCode .python}
    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=run_config,
    ```

    to

    ``` {.sourceCode .python}
    classifier = SelectedEstimator(
        model_fn=model_function, model_dir=model_dir, config=run_config,
    ```

54. Changed

    ``` {.sourceCode .python}
    'loss_scale': flags_core.get_loss_scale(flags_obj),
    ```

    to

    ``` {.sourceCode .python}
    'model_type': flags_obj.model_type,
    'loss_scale': flags_core.get_loss_scale(flags_obj,
                                            default_for_fp16=128),
    ```

55. Changed

    ``` {.sourceCode .python}
    'fine_tune': flags_obj.fine_tune
    ```

    to

    ``` {.sourceCode .python}
    'fine_tune': flags_obj.fine_tune,
    'num_workers': num_workers,
    'train_epochs': flags_obj.train_epochs,
    'warmup_epochs': flags_obj.warmup_epochs,
    'use_cosine_lr': flags_obj.use_cosine_lr,
    ```

56. Added

    ``` {.sourceCode .python}
    'model_type': flags_obj.model_type,
    ```

57. Added

    ``` {.sourceCode .python}
    'num_workers': num_workers,
    ```

58. Changed

    ``` {.sourceCode .python}
    model_dir=flags_obj.model_dir,
    ```

    to

    ``` {.sourceCode .python}
    model_dir=model_dir,
    ```

59. Changed

    ``` {.sourceCode .python}
    def input_fn_train(num_epochs):
    ```

    to

    ``` {.sourceCode .python}
    if flags.FLAGS.is_mlperf_enabled:
      _log_cache = []
      def formatter(x):
        """Abuse side effects to get tensors out of the model_fn."""
        if _log_cache:
          _log_cache.pop()
          _log_cache.append(x.copy())
          return str(x)

      compliance_hook = tf.estimator.LoggingTensorHook(
        tensors={_NUM_EXAMPLES_NAME: _NUM_EXAMPLES_NAME},
        every_n_iter=int(1e10),
        at_end=True,
        formatter=formatter)
    else:
      compliance_hook = None

    if horovod_enabled():

      if "tf_profiler_hook" not in flags_obj.hooks and os.environ.get("TF_RANGE_TRACE", False):
        from TensorFlow.common.utils import RangeTFProfilerHook
        begin = (imagenet_main.NUM_IMAGES["train"] // (flags_obj.batch_size * hvd.size()) + 100)
        train_hooks.append(RangeTFProfilerHook(begin,20, "./rank-{}".format(hvd.rank())))

      if "synapse_logger_hook" not in flags_obj.hooks and "range" == os.environ.get("HABANA_SYNAPSE_LOGGER", "False").lower():
        from TensorFlow.common.horovod_helpers import SynapseLoggerHook
        begin = (imagenet_main.NUM_IMAGES["train"] // (flags_obj.batch_size * hvd.size()) + 100)
        end = begin + 100
        print("Begin: {}".format(begin))
        print("End: {}".format(end))
        train_hooks.append(SynapseLoggerHook(list(range(begin, end)), False))
      train_hooks.append(hvd.BroadcastGlobalVariablesHook(0))


    def input_fn_train(num_epochs, input_context=None):
    ```

60. Changed

    ``` {.sourceCode .python}
    batch_size=distribution_utils.per_device_batch_size(
    ```

    to

    ``` {.sourceCode .python}
    batch_size=distribution_utils.per_replica_batch_size(
    ```

61. Changed

    ``` {.sourceCode .python}
    dtype=flags_core.get_tf_dtype(flags_obj),
    ```

    to

    ``` {.sourceCode .python}
    dtype=flags_core.get_dl_type(flags_obj),
    ```

62. Changed

    ``` {.sourceCode .python}
    num_parallel_batches=flags_obj.datasets_num_parallel_batches)
    ```

    to

    ``` {.sourceCode .python}
    input_context=input_context, experimental_preloading=experimental_preloading)
    ```

63. Changed

    ``` {.sourceCode .python}
    batch_size=distribution_utils.per_device_batch_size(
    ```

    to

    ``` {.sourceCode .python}
    batch_size=distribution_utils.per_replica_batch_size(
    ```

64. Changed

    ``` {.sourceCode .python}
    dtype=flags_core.get_tf_dtype(flags_obj))
    ```

    to

    ``` {.sourceCode .python}
        dtype=flags_core.get_dl_type(flags_obj), experimental_preloading=experimental_preloading)
      train_epochs = (0 if flags_obj.eval_only or not flags_obj.train_epochs else
                      flags_obj.train_epochs)
    ```

65. Changed

    ``` {.sourceCode .python}
    if flags_obj.eval_only or not flags_obj.train_epochs:
      # If --eval_only is set, perform a single loop with zero train epochs.
      schedule, n_loops = [0], 1
    ```

    to

    ``` {.sourceCode .python}
    max_train_steps = flags_obj.max_train_steps
    global_batch_size = flags_obj.batch_size * (hvd.size() if horovod_enabled() else 1)
    steps_per_epoch = (imagenet_main.NUM_IMAGES['train'] // global_batch_size)
    if max_train_steps is None:
      max_train_steps = steps_per_epoch * train_epochs

    max_eval_steps = flags_obj.max_eval_steps
    if max_eval_steps is None:
      max_eval_steps = (imagenet_main.NUM_IMAGES['validation'] + flags_obj.batch_size - 1) // flags_obj.batch_size

    use_train_and_evaluate = flags_obj.use_train_and_evaluate or num_workers > 1
    if use_train_and_evaluate:
      train_spec = tf.estimator.TrainSpec(
          input_fn=lambda input_context=None: input_fn_train(
              train_epochs, input_context=input_context),
          hooks=train_hooks,
          max_steps=max_train_steps)
      eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval)
      tf.compat.v1.logging.info('Starting to train and evaluate.')
      tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
      # tf.estimator.train_and_evalute doesn't return anything in multi-worker
      # case.
      eval_results = {}
    ```

66. Changed

    ``` {.sourceCode .python}
        # Compute the number of times to loop while training. All but the last
        # pass will train for `epochs_between_evals` epochs, while the last will
        # train for the number needed to reach `training_epochs`. For instance if
        #   train_epochs = 25 and epochs_between_evals = 10
        # schedule will be set to [10, 10, 5]. That is to say, the loop will:
        #   Train for 10 epochs and then evaluate.
        #   Train for another 10 epochs and then evaluate.
        #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
        n_loops = math.ceil(flags_obj.train_epochs / flags_obj.epochs_between_evals)
        schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
        schedule[-1] = flags_obj.train_epochs - sum(schedule[:-1])  # over counting.
      for cycle_index, num_train_epochs in enumerate(schedule):
        tf.logging.info('Starting cycle: %d/%d', cycle_index, int(n_loops))
        if num_train_epochs:
          classifier.train(input_fn=lambda: input_fn_train(num_train_epochs),
                           hooks=train_hooks, max_steps=flags_obj.max_train_steps)
        tf.logging.info('Starting to evaluate.')
        # flags_obj.max_train_steps is generally associated with testing and
        # profiling. As a result it is frequently called with synthetic data, which
        # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
        # eval (which is generally unimportant in those circumstances) to terminate.
        # Note that eval will run for max_train_steps each loop, regardless of the
        # global_step count.
        eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                           steps=flags_obj.max_train_steps)
        benchmark_logger.log_evaluation_result(eval_results)
        if model_helpers.past_stop_threshold(
            flags_obj.stop_threshold, eval_results['accuracy']):
          break
    ```

    to

    ``` {.sourceCode .python}
    if train_epochs == 0:
      # If --eval_only is set, perform a single loop with zero train epochs.
      schedule, n_loops = [0], 1
    else:
      # Compute the number of times to loop while training. All but the last
      # pass will train for `epochs_between_evals` epochs, while the last will
      # train for the number needed to reach `training_epochs`. For instance if
      #   train_epochs = 25 and epochs_between_evals = 10
      # schedule will be set to [10, 10, 5]. That is to say, the loop will:
      #   Train for 10 epochs and then evaluate.
      #   Train for another 10 epochs and then evaluate.
      #   Train for a final 5 epochs (to reach 25 epochs) and then evaluate.
      n_loops = math.ceil(train_epochs / flags_obj.epochs_between_evals)
      schedule = [flags_obj.epochs_between_evals for _ in range(int(n_loops))]
      schedule[-1] = train_epochs - sum(schedule[:-1])  # over counting.

    if flags.FLAGS.is_mlperf_enabled:
      mllogger.event(key=mllog.constants.CACHE_CLEAR)
      mllogger.start(key=mllog.constants.RUN_START)
      mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE,
                     value=flags_obj.batch_size*num_workers)

    final_step = 0

    if flags.FLAGS.is_mlperf_enabled:
      success = False
      if flags_obj.train_offset > 0:
        final_step += flags_obj.train_offset * steps_per_epoch
        classifier.train(
              input_fn=lambda input_context=None: input_fn_train(
              flags_obj.train_offset, input_context=input_context),
              hooks=train_hooks + [compliance_hook],
              max_steps=max_train_steps if max_train_steps < final_step else final_step)
        mllogger.event(key=mllog.constants.FIRST_EPOCH_NUM, value=0,metadata={'number of epochs before main loop: ': flags_obj.train_offset})

    for cycle_index, num_train_epochs in enumerate(schedule):
      tf.compat.v1.logging.info('Starting cycle: %d/%d', cycle_index,
                                int(n_loops))
      if flags.FLAGS.is_mlperf_enabled:
        mllogger.start(key=mllog.constants.BLOCK_START, value=cycle_index+1)
        mllogger.event(key=mllog.constants.FIRST_EPOCH_NUM, value=cycle_index*flags_obj.epochs_between_evals + flags_obj.train_offset)
        mllogger.event(key=mllog.constants.EPOCH_COUNT, value=flags_obj.epochs_between_evals)

        for j in range(flags_obj.epochs_between_evals):
          mllogger.event(key=mllog.constants.EPOCH_NUM,
                         value=cycle_index  * flags_obj.epochs_between_evals + j +  flags_obj.train_offset)

      if num_train_epochs:
        # Since we are calling classifier.train immediately in each loop, the
        # value of num_train_epochs in the lambda function will not be changed
        # before it is used. So it is safe to ignore the pylint error here
        # pylint: disable=cell-var-from-loop
        final_step += num_train_epochs * steps_per_epoch
        classifier.train(
            input_fn=lambda input_context=None: input_fn_train(
                num_train_epochs, input_context=input_context),
            hooks=train_hooks + [compliance_hook] if compliance_hook is not None else train_hooks,
            max_steps=max_train_steps if max_train_steps < final_step else final_step)
        if flags.FLAGS.is_mlperf_enabled:
            mllogger.end(key=mllog.constants.BLOCK_STOP, value=cycle_index+1)

      if flags.FLAGS.is_mlperf_enabled:
        mllogger.start(key=mllog.constants.EVAL_START)
      # max_eval_steps is associated with testing and profiling.
      # As a result it is frequently called with synthetic data,
      # which will iterate forever. Passing steps=max_eval_steps
      # allows the eval (which is generally unimportant in those circumstances)
      # to terminate. Note that eval will run for max_eval_steps each loop,
      # regardless of the global_step count.
      if flags_obj.get_flag_value("return_before_eval", False):
        return {}
      if flags_obj.get_flag_value("disable_eval", False):
        eval_results = None
        continue
      tf.compat.v1.logging.info('Starting to evaluate.')
      eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                         steps=max_eval_steps)

      if flags.FLAGS.is_mlperf_enabled:
        mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=int(eval_results[_NUM_EXAMPLES_NAME]))
        valdiation_epoch = (cycle_index + 1) * flags_obj.epochs_between_evals - 1 + flags_obj.train_offset
        mllogger.event(key=mllog.constants.EVAL_ACCURACY, value=float(eval_results['accuracy']),metadata={'epoch_num: ': valdiation_epoch})
        mllogger.end(key=mllog.constants.EVAL_STOP,metadata={'epoch_num: ' : valdiation_epoch})

      benchmark_logger.log_evaluation_result(eval_results)

      if flags_obj.stop_threshold:
        if horovod_enabled():
          past_treshold = tf.cast(model_helpers.past_stop_threshold(
              flags_obj.stop_threshold, eval_results['accuracy']), tf.float32)
          global_past_treshold = tf.math.greater(
              hvd.allreduce(past_treshold, op=hvd.Sum), tf.zeros(1, tf.float32))
          if global_past_treshold.eval(session=tf.compat.v1.Session()):
            break
        else:
          if model_helpers.past_stop_threshold(
              flags_obj.stop_threshold, eval_results['accuracy']):
            break
    ```

67. Removed

    ``` {.sourceCode .python}
    return eval_results
    ```

68. Changed

    ``` {.sourceCode .python}
    def define_resnet_flags(resnet_size_choices=None):
    ```

    to

    ``` {.sourceCode .python}
    stats = {}
    stats['eval_results'] = eval_results
    stats['train_hooks'] = train_hooks

    if flags.FLAGS.is_mlperf_enabled:
      mllogger.event(key=mllog.constants.RUN_STOP, value={"success": success})
      mllogger.end(key=mllog.constants.RUN_STOP)

    return stats

    def define\_resnet\_flags(resnet\_size\_choices=None, dynamic\_loss\_scale=False,

       fp16\_implementation=False):
    ```

69. Changed

    ``` {.sourceCode .python}
    flags_core.define_base()
    ```

    to

    ``` {.sourceCode .python}
    flags_core.define_base(clean=True, train_epochs=True,
                           epochs_between_evals=True, stop_threshold=True,
                           num_gpu=True, hooks=True, export_dir=True,
                           distribution_strategy=True)
    ```

70. Added

    ``` {.sourceCode .python}
    inter_op=True,
    intra_op=True,
    synthetic_data=True,
    dtype=True,
    data_loader_image_type = True,
    all_reduce_alg=True,
    num_packs=True,
    ```

71. Changed

    ``` {.sourceCode .python}
    datasets_num_parallel_batches=True)
    ```

    to

    ``` {.sourceCode .python}
    dynamic_loss_scale=dynamic_loss_scale,
    fp16_implementation=fp16_implementation,
    loss_scale=True,
    tf_data_experimental_slack=True,
    max_train_steps=True,
    max_eval_steps=True)
    ```

72. Added

    ``` {.sourceCode .python}
    flags_core.define_distribution()
    flags_core.define_experimental()
    ```

73. Added

    ``` {.sourceCode .python}
    flags.DEFINE_integer(
        name='train_offset', default=0,
        help='Number of train epochs before the train loop - for MLPER.')
    flags.DEFINE_bool(
        name='is_mlperf_enabled', short_name='mlperf_enable', default=False,
        help=flags_core.help_wrap(
            'IF true enable mlperf logger and mlperf lars optimizer config'))
    ```

74. Added

    ``` {.sourceCode .python}
    name='return_before_eval', short_name='rbv', default=False,
    help=flags_core.help_wrap(
        'If True exit training before evaluation (for testing)'))

        name='return_before_eval', short_name='rbv', default=False,
          help=flags_core.help_wrap(
              'If True exit training before evaluation (for testing)'))
      flags.DEFINE_bool(
          name='disable_eval', default=False,
          help=flags_core.help_wrap(
              'If True disables evaluation step after each training epoch'))
      flags.DEFINE_bool(
    ```

75. Changed

    ``` {.sourceCode .python}
    name='turn_off_distribution_strategy', default=False,
    help=flags_core.help_wrap('Set to True to not use distribution '
                              'strategies.'))
    ```

    to

    ``` {.sourceCode .python}
          name='use_train_and_evaluate', default=False,
          help=flags_core.help_wrap(
              'If True, uses `tf.estimator.train_and_evaluate` for the training '
              'and evaluation loop, instead of separate calls to `classifier.train '
              'and `classifier.evaluate`, which is the default behavior.'))
      flags.DEFINE_bool(
          name='enable_lars', default=False,
          help=flags_core.help_wrap(
              'Enable LARS optimizer for large batch training.'))
      flags.DEFINE_float(
          name='label_smoothing', default=0.0,
          help=flags_core.help_wrap(
              'Label smoothing parameter used in the softmax_cross_entropy'))
      flags.DEFINE_integer(
          name='epochs_for_lars', default=38,
          help=flags_core.help_wrap(
              'Epochs used in lars optimizer calculations for decay steps.'))
      flags.DEFINE_integer(
          name='warmup_epochs', default=3,
          help=flags_core.help_wrap(
              'The number of epochs used for LR warmup'))
      flags.DEFINE_float(
          name='start_learning_rate', default=9.5,
          help=flags_core.help_wrap(
              'Learning rate used in lars optimizer calculations for decay steps.'))
      flags.DEFINE_float(
          name='weight_decay', default=1e-4,
          help=flags_core.help_wrap(
              'Weight decay coefficiant for l2 regularization.'))
      flags.DEFINE_bool(
          name='disable_warmup', default=False,
          help=flags_core.help_wrap(
              'If True disable warmup phase for learning rate.'))
      flags.DEFINE_bool(
          name='use_cosine_lr', default=False,
          help=flags_core.help_wrap(
              'Use cosine decay learning rate schedule'))
      flags.DEFINE_float(
          name='momentum', default=0.9,
          help=flags_core.help_wrap(
              'Momentum value used for optimization.'))
      flags.DEFINE_enum(
          name='model_type', default='resnet',
          enum_values=['resnet', 'resnext'],
          help=flags_core.help_wrap(
              'Residual model type.'))
    ```
