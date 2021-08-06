---
title: Efficientdet changes
---

Scripts were taken from
<https://gerrit.habana-labs.com/plugins/gitiles/workloads/+/master/Vision/Recognition/ObjectDetection/auto_ml-horovod+tf2.2/efficientdet/https://gerrit.habana-labs.com/plugins/gitiles/workloads/+/master/Vision/Recognition/ObjectDetection/auto_ml-horovod+tf2.2/efficientdet/>

Files used:

-   dataloader.py
-   det\_model\_fn.py
-   efficientdet\_arch.py
-   hooks.py
-   \_\_[init](init__.py).py
-   utis.py
-   main.py
-   normalization\_v2.py
-   utils.py

Main changes are:

1.  Added HPU support
2.  Added Horovod support for multinode
3.  Changed default values for some parameters
4.  Added new parameters
5.  Added new metrics for TensorBoard

dataloader.py
=============

1.  Changed

    ``` {.sourceCode .python}
    from horovod_estimator import hvd
    ```

    to

    ``` {.sourceCode .python}
    from TensorFlow.common.horovod_helpers import hvd, horovod_enabled
    ```

2.  Changed

    ``` {.sourceCode .python}
    if hvd is not None:
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
    ```

det\_model\_fn.py
=================

1.  Changed

    ``` {.sourceCode .python}
    from horovod_estimator import show_model, hvd
    ```

    to

    ``` {.sourceCode .python}
    from horovod_estimator import show_model, hvd, horovod_enabled
    ```

2.  Changed

    ``` {.sourceCode .python}
    if not isinstance(optimizer, hvd._DistributedOptimizer):
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
    ```

3.  Changed

    ``` {.sourceCode .python}
    init_weights_hook = BroadcastGlobalVariablesHook(root_rank=0, model_dir=params['model_dir'])
    training_hooks = [init_weights_hook]
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
      init_weights_hook = BroadcastGlobalVariablesHook(root_rank=0, model_dir=params['model_dir'])
      training_hooks = [init_weights_hook]
    else:
      training_hooks = []
    ```

4.  Changed

    ``` {.sourceCode .python}
    logging_hook = LoggingTensorHook(dict(utils.summaries), summary_dir=params['model_dir'])
    ```

    to

    ``` {.sourceCode .python}
    logging_hook = LoggingTensorHook(dict(utils.summaries), summary_dir=params['model_dir'],
                                     every_n_iter=params['every_n_iter'])
    ```

    To allow logging after variable number of iterations set by
    parameter.

5.  Added

    ``` {.sourceCode .python}
    from TensorFlow.common.tb_utils import ExamplesPerSecondEstimatorHook
    ```

6.  Added

    ``` {.sourceCode .python}
    utils.scalar('loss', total_loss)  # for consistency
    ```

7. Changed

    ``` {.sourceCode .python}
    logging_hook = LoggingTensorHook(dict(utils.summaries), summary_dir=params['model_dir'],
                                     every_n_iter=params['every_n_iter'])
    training_hooks.append(logging_hook)
    ```

    to

    ``` {.sourceCode .python}
    training_hooks.extend([
      LoggingTensorHook(
        dict(utils.summaries), summary_dir=params['model_dir'],
        every_n_iter=params['every_n_iter']),
      ExamplesPerSecondEstimatorHook(
        params['batch_size'], every_n_steps=params['every_n_iter'],
        output_dir=params['model_dir'])
    ])
    ```

    Logs examples/sec into TensorBoard.

efficientdet\_arch.py
=====================

1.  Changed

    ``` {.sourceCode .python}
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    ```

    to

    ``` {.sourceCode .python}
    #
    # 6D tensors are currently not supported
    # TODO: SW-35724 revert change when they are supported
    data = tf.reshape(data, [bs * h, 1, w, 1, c]) * tf.ones(
        [1 * 1, scale, 1, scale, 1], dtype=data.dtype)
    ```

    Reducing 6D tensor to 5D as 6D are not currently supported by HW.

hooks.py
========

1.  Changed

    ``` {.sourceCode .python}
    try:
        import horovod.tensorflow as hvd
    except ImportError:
        hvd = None
    ```

    to

    ``` {.sourceCode .python}
    from TensorFlow.common.horovod_helpers import hvd
    ```

2.  Removed

    ``` {.sourceCode .python}
    import matplotlib as mpl
    ```

3.  Removed

    ``` {.sourceCode .python}
    from sklearn.metrics import confusion_matrix
    ```

4.  Removed

    ``` {.sourceCode .python}
    mpl.use('Agg')
    ```

5.  Changed

    ``` {.sourceCode .python}
    def __init__(self, root_rank, pretrained_model_path=None, exclusions=[], device='', model_dir=None):
    ```

    to

    ``` {.sourceCode .python}
    def __init__(self, root_rank, pretrained_model_path=None, exclusions=[], device='/device:HPU:0', model_dir=None):
    ```

    For running topology on HPU.

6.  Changed

    ``` {.sourceCode .python}
    this_summary.value.add(tag='avg/{}'.format(tag), simple_value=value)
    ```

    to

    ``` {.sourceCode .python}
    this_summary.value.add(tag=tag, simple_value=value)
    ```

------------------------------------------------------------------------

1.  Changed

    ``` {.sourceCode .python}
    try:
        import horovod.tensorflow as hvd
    except ImportError:
        hvd = None
    ```

    to

    ``` {.sourceCode .python}
    from TensorFlow.common.horovod_helpers import hvd, horovod_enabled
    ```

utis.py
=======

1.  Changed

    ``` {.sourceCode .python}
    try:
        import horovod.tensorflow as hvd
    except ImportError:
        hvd = None
    ```

    to

    ``` {.sourceCode .python}
    from TensorFlow.common.horovod_helpers import hvd, hvd_init, horovod_enabled
    ```

2.  Changed

    ``` {.sourceCode .python}
    if hvd is not None:
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
    ```

3.  Changed

    ``` {.sourceCode .python}
    if not IS_HVD_INIT and hvd is not None:
        hvd.init()
    ```

    to

    ``` {.sourceCode .python}
    if not IS_HVD_INIT and horovod_enabled():
        hvd_init()
    ```

4.  Changed

    ``` {.sourceCode .python}
    if hvd is not None:
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
    ```

5.  Changed

    ``` {.sourceCode .python}
    if hvd is not None:
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
    ```

main.py
=======

1.  Changed

    ``` {.sourceCode .python}
    from horovod_estimator import HorovodEstimator, hvd_try_init, hvd_info_rank0, hvd
    ```

    to

    ``` {.sourceCode .python}
    from horovod_estimator import HorovodEstimator, hvd_try_init, hvd_info_rank0, hvd, horovod_enabled
    ```

2.  Added

    ``` {.sourceCode .python}
    from TensorFlow.common import library_loader
    library_loader.load_habana_module()
    ```

    For HPU support.

3.  Changed

    ``` {.sourceCode .python}
    flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs/GPUs')
    ```

    to

    ``` {.sourceCode .python}
    flags.DEFINE_bool('use_tpu', False, 'Use TPUs rather than CPUs/GPUs')
    ```

    Changed default value to compiant with HPU.

4.  Changed

    ``` {.sourceCode .python}
    flags.DEFINE_string('model_dir', None, 'Location of model_dir')
    ```

    to

    ``` {.sourceCode .python}
    flags.DEFINE_string('model_dir', 'train_model_dir', 'Location of model_dir')
    ```

    Set default value.

5.  Changed

    ``` {.sourceCode .python}
    flags.DEFINE_string('hparams', '',
    ```

    to

    ``` {.sourceCode .python}
    flags.DEFINE_string('hparams', 'num_classes=91,use_bfloat16=false',
    ```

    Set default value to compiant with HPU.

6.  Changed

    ``` {.sourceCode .python}
    flags.DEFINE_string('model_name', 'efficientdet-d1',
    ```

    to

    ``` {.sourceCode .python}
    flags.DEFINE_string('model_name', 'efficientdet-d0',
    ```

    Changed default model variant to d0.

7.  Added

    ``` {.sourceCode .python}
    flags.DEFINE_integer('log_every_n_steps', 100, 'Number of iterations after which '
                         'training parameters are logged.')
    flags.DEFINE_integer('cp_every_n_steps', 600, 'Number of iterations after which '
                         'checkpoint is saved.')
    ```

    Added parameters for logging and saving checkpoints every certain
    number of steps.

8.  Changed

    ``` {.sourceCode .python}
    os.environ['TF_GPU_THREAD_COUNT'] = '1' if hvd is None else str(hvd.size())
    ```

    to

    ``` {.sourceCode .python}
    os.environ['TF_GPU_THREAD_COUNT'] = '1' if not horovod_enabled() else str(hvd.size())
    ```

9.  Changed

    ``` {.sourceCode .python}
    if hvd is not None:
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
    ```

10. Changed

    ``` {.sourceCode .python}
    if hvd is not None:
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
    ```

11. Changed

    ``` {.sourceCode .python}
    if hvd is not None:
        num_shards = hvd.size()
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
        num_shards = hvd.size()
    else:
        num_shards = 1
    ```

12. Changed

    ``` {.sourceCode .python}
    save_checkpoints_steps=600)
    ```

    to

    ``` {.sourceCode .python}
    save_checkpoints_steps=FLAGS.cp_every_n_steps)
    ```

    To allow saving checkpoints after variable number of iterations set
    by parameter.

13. Removed

    ``` {.sourceCode .python}
    # train_estimator = tf.estimator.tpu.TPUEstimator(
    #     model_fn=model_fn_instance,
    #     use_tpu=FLAGS.use_tpu,
    #     train_batch_size=FLAGS.train_batch_size,
    #     config=run_config,
    #     params=params)
    ```

    The code was commented out.

14. Added

    ``` {.sourceCode .python}
    params['every_n_iter'] = FLAGS.log_every_n_steps
    ```

    To allow logging after variable number of iterations set by
    parameter.

15. Removed

    ``` {.sourceCode .python}
    # if FLAGS.eval_after_training:
    #   # Run evaluation after training finishes.
    #   eval_params = dict(
    #       params,
    #       use_tpu=False,
    #       input_rand_hflip=False,
    #       is_training_bn=False,
    #       use_bfloat16=False,
    #   )
    #   eval_estimator = tf.estimator.tpu.TPUEstimator(
    #       model_fn=model_fn_instance,
    #       use_tpu=False,
    #       train_batch_size=FLAGS.train_batch_size,
    #       eval_batch_size=FLAGS.eval_batch_size,
    #       config=run_config,
    #       params=eval_params)
    #   eval_results = eval_estimator.evaluate(
    #       input_fn=dataloader.InputReader(FLAGS.validation_file_pattern,
    #                                       is_training=False),
    #       steps=FLAGS.eval_samples//FLAGS.eval_batch_size)
    #   logging.info('Eval results: %s', eval_results)
    #   ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
    #   utils.archive_ckpt(eval_results, eval_results['AP'], ckpt)
    ```

    The code was commented out.


16.  Added

    ``` {.sourceCode .python}
    from TensorFlow.common.tb_utils import write_hparams_v1
    ```

17.  Added

    ``` {.sourceCode .python}
    write_hparams_v1(FLAGS.model_dir, FLAGS.flag_values_dict())
    ```

normalization\_v2.py
====================

1.  Changed

    ``` {.sourceCode .python}
    from horovod_estimator import hvd, hvd_info
    ```

    to

    ``` {.sourceCode .python}
    from TensorFlow.common.horovod_helpers import hvd, horovod_enabled
    ```

2.  Changed

    ``` {.sourceCode .python}
    if hvd is not None:
    ```

    to

    ``` {.sourceCode .python}
    if horovod_enabled():
    ```

utils.py
========

1.  Changed

    ``` {.sourceCode .python}
    from horovod_estimator import hvd
    ```

    to

    ``` {.sourceCode .python}
    from TensorFlow.common.horovod_helpers import hvd, horovod_enabled
    ```
