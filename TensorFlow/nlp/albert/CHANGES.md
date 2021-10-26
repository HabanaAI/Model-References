# ALBERT changes

Originally, scripts were taken from [Google Github](https://github.com/google-research/albert)

Files used:
* modeling.py
* optimization.py
* run_classifier.py
* run_pretraining.py
* run_squad_v1.py
* tokenization.py

All of above files were converted to TF2 by using [tf_upgrade_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool. Additionally, some other changes were committed for specific files. Also, in these files, horovod usage was changed to use horovod functions wrappers and global `hvd` object instead of passing `hvd` as an argument:

Added file:
. custom_layer_norm.py

## modeling.py
* In layer_normal function changed:
  ```python
   def layer_norm(input_tensor, name=None):
   ...
     return contrib_layers.layer_norm(
         inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
  ```
to:
  ```python
   def layer_norm(input_tensor, name=None):
   ...
     return custom_layer_norm(input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name) ...
  ```
## optimization.py
Added
```python
from TensorFlow.common.horovod_helpers import hvd, horovod_enabled
```

Added for horovod
```python
  if horovod_enabled():
    optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True)
```
Changed
```python
  if use_tpu:
    optimizer = contrib_tpu.CrossShardOptimizer(optimizer)
```
to
```python
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)
```

Changed
```python
  grads = tf.gradients(
      loss, tvars, colocate_gradients_with_ops=colocate_gradients_with_ops)
```
to
```python
  grads_and_vars = optimizer.compute_gradients(loss, tvars, gate_gradients=tf.compat.v1.train.Optimizer.GATE_NONE)
  grads = [grad for grad,var in grads_and_vars]
  tvars = [var for grad,var in grads_and_vars]
```

Changed
```python
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
```
to
```python
  # To prevent warnings casued by race condition, control edge was added
  # More info: https://github.com/google-research/bert/issues/1050
  with tf.control_dependencies([global_step.assign(new_global_step)]):
    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)
```

## run_pretraining.py
Changed:
```python
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import tpu as contrib_tpu
```
to:
```python
from TensorFlow.common.library_loader import load_habana_modul
from TensorFlow.common.horovod_helpers import hvd, hvd_init, hvd_size, hvd_rank, horovod_enabled
```
Added:
```python
flags.DEFINE_bool(
    "use_einsum", False,
    "Whether to use tf.einsum or tf.reshape+tf.matmul for dense layers. Must "
    "be set to False for TFLite compatibility.")

flags.DEFINE_bool("use_horovod", False, "Whether to use Horovod for multi-gpu runs")
```

Changed:
```python
  def model_fn_builder(albert_config, init_checkpoint, learning_rate,
                                           num_train_steps, num_warmup_steps, use_tpu,
                                           use_one_hot_embeddings, optimizer, poly_power,
                                           start_warmup_step)
                                           ...
    model = modeling.AlbertModel(
            config=albert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)
            ...
```
to:
```python
  def model_fn_builder(albert_config, init_checkpoint, learning_rate,
                                           num_train_steps, num_warmup_steps, use_tpu,
                                           use_one_hot_embeddings, optimizer, poly_power,
                                           start_warmup_step, use_einsum):
                                           ...
    model = modeling.AlbertModel(
            config=albert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            use_einsum=use_einsum)
            ...
```

Inside def input_fn_builder function
Added:
```python
      ...
      if horovod_enabled():
        d = d.shard(hvd_size(), hvd_rank())
```

Inside def main(_) function
Added:
```python
  def main(_):
    ...
    if horovod_enabled():
      FLAGS.output_dir = FLAGS.output_dir if hvd_rank() == 0 else os.path.join(FLAGS.output_dir, str(hvd_rank()))
```
Added:
```python
  def main(_):
    ...
    if FLAGS.use_horovod and len(input_files) < hvd.size():
      input_files = [input_files[0] for i in range(hvd.size())]
```
Added:
```python
  def main(_):
    ...
    num_train_steps = FLAGS.num_train_steps
    num_warmup_steps = FLAGS.num_warmup_steps
    if FLAGS.do_train and horovod_enabled():
      num_train_steps //= hvd_size()
      num_warmup_steps //= hvd_size()
```
Changed:
```python
  def main(_):
     ...
     model_fn = model_fn_builder(
     albert_config=albert_config,
     init_checkpoint=FLAGS.init_checkpoint,
     learning_rate=FLAGS.learning_rate,
     num_train_steps=FLAGS.num_train_steps,
     num_warmup_steps=FLAGS.num_warmup_steps,
     use_tpu=FLAGS.use_tpu,
     use_one_hot_embeddings=FLAGS.use_tpu,
     optimizer=FLAGS.optimizer,
     poly_power=FLAGS.poly_power,
     start_warmup_step=FLAGS.start_warmup_step)
```
to:
```python
  def main(_):
    ...
    model_fn = model_fn_builder(
    albert_config=albert_config,
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate if not FLAGS.use_horovod else FLAGS.learning_rate*hvd_size(),
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_tpu,
    optimizer=FLAGS.optimizer,
    poly_power=FLAGS.poly_power,
    start_warmup_step=FLAGS.start_warmup_step,
    use_einsum=FLAGS.use_einsum)
```
Added:
```python
  def main(_):
    ...
    if FLAGS.do_train:
      training_hooks = []
      if horovod_enabled():
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
```
Changed:
```python
  def main(_):
    ...
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps
    if FLAGS.do_eval:
      tf.logging.info("***** Running evaluation *****")
```
to:
```python
  def main(_):
    ...
    estimator.train(input_fn=train_input_fn, hooks=training_hooks, max_steps=FLAGS.num_train_steps)
    if FLAGS.do_eval and (not FLAGS.use_horovod or hvd_rank() == 0):
      tf.logging.info("***** Running evaluation *****")
```
Inside if __name__ == "__main__":
Added:
```python
  if FLAGS.use_horovod:
    hvd_init()
  log_info_devices = load_habana_module()
  tf.logging.info("Devices:\n%s", log_info_devices)
```

## run_squad_v1.py
* Overall changed from contrib_tpu to tf.estimator.tpu from TF2 update
* Added flags --use_horovod
* Added loading of Habana library
* Added horovod support with our custom hvd functions wrappers (`hvd_XXX` instead of `hvd.XXX`)
   * Initialize hvd
   * Adjust train_batch_size:
   ```python
      train_batch_size = FLAGS.train_batch_size
      if horovod_enabled():
        train_batch_size = train_batch_size * hvd.size()
   ```
   * Added `hvd.BroadcastGlobalVariablesHook(0)` to estimator
   * MultiHLS support
   * Added:
   ```python
      def main(_):
        ...
        start_index = 0
        end_index = len(train_examples)
        per_worker_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]
        worker_id = 0
        if horovod_enabled():
          per_worker_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record_{}".format(i)) for i in range(hvd.local_size())]
          num_examples_per_rank = len(train_examples) // hvd.size()
          remainder = len(train_examples) % hvd.size()
          worker_id = hvd.rank()
          if worker_id < remainder:
            start_index = worker_id * (num_examples_per_rank+1)
            end_index = start_index + num_examples_per_rank + 1
          else:
            start_index = worker_id * num_examples_per_rank + remainder
            end_index = start_index + (num_examples_per_rank)
        learning_rate = FLAGS.learning_rate
        if horovod_enabled():
          learning_rate = learning_rate * hvd.size()
        ...
   ```

* Added ScopedAllocator support by changing:
  ```python
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=model_dir,
    keep_checkpoint_max=0,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))
   ```
   to
   ```python
    # The Scoped Allocator Optimization is enabled by default unless disabled by a flag.
    if condition_env_var('TF_DISABLE_SCOPED_ALLOCATOR', default=False):
      session_config = None
    else:
      from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=import-error
      session_config = tf.compat.v1.ConfigProto()
      session_config.graph_options.rewrite_options.scoped_allocator_optimization = rewriter_config_pb2.RewriterConfig.ON
      enable_op = session_config.graph_options.rewrite_options.scoped_allocator_opts.enable_op
      del enable_op[:]
      enable_op.append("HorovodAllreduce")
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        ...
        session_config=session_config)
   ```

* Added train.tf_record load code, this is related to splitting tfrecords based on multiple nodes
   ```python
  start_index = 0
  end_index = len(train_examples)
  per_worker_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record")]
  worker_id = 0

  if horovod_enabled():
    per_worker_filenames = [os.path.join(FLAGS.output_dir, "train.tf_record_{}".format(i)) for i in range(hvd.local_size())]
    num_examples_per_rank = len(train_examples) // hvd.size()
    remainder = len(train_examples) % hvd.size()
    worker_id = hvd.rank()
    if worker_id < remainder:
      start_index = worker_id * (num_examples_per_rank + 1)
      end_index = start_index + num_examples_per_rank + 1
    else:
      start_index = worker_id * num_examples_per_rank + remainder
      end_index = start_index + (num_examples_per_rank)

  learning_rate = FLAGS.learning_rate
```

* Added ``SynapseLoggerHook``:
   ```python
    if "range" == os.environ.get("HABANA_SYNAPSE_LOGGER", "False").lower():
      from demo.horovod_helpers import SynapseLoggerHook
      begin = 670
      end = begin + 10
      print("Begin: {}".format(begin))
      print("End: {}".format(end))
      train_hooks.append(SynapseLoggerHook(list(range(begin, end)), False))
   ```
* Added `PerfLoggingHook`

## run_classifier.py
* Added flags --use_horovod
* Some minor code refatoring:
   * Changed import statements to fit our repo directory structure
* Added loading of Habana library
* Added horovod support, similarly as in run_squad.py except for MultiHLS support
* Added ScopedAllocator support, as in run_squad_v1.py
* Added `SynapseLoggerHook`, as in run_squad_v1.py
* Added``PerfLoggingHook`
