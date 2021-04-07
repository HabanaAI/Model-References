# BERT Finetuning changes

Originally, scripts were taken from [Google Github](https://github.com/google-research/bert), with commit hash [cc7051dc](https://github.com/google-research/bert/tree/cc7051dc592802f501e8a6f71f8fb3cf9de95dc9).

Files used:
* modeling.py
* optimization.py
* run_classifier.py
* run_classifier_with_tfhub.py
* run_squad.py
* tokenization.py

All of above files were converted to TF2 by using [tf_upgrade_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool. Additionally, some other changes were committed for specific files.

# optimization.py

Changed:
```python
    def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
 
    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
       train_op = optimizer.apply_gradients(
           zip(grads, tvars), global_step=global_step)
       # Normally the global step update is done inside of `apply_gradients`.
       # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
       # a different optimizer, you should probably take this line out.
       new_global_step = global_step + 1
       train_op = tf.group(train_op, [global_step.assign(new_global_step)])
       return train_op
 ```
 to:
 ```python
    def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
       
       # This is how the model was pre-trained.
       (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
       # Normally the global step update is done inside of `apply_gradients`.
       # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
       # a different optimizer, you should probably take this line out.
       new_global_step = global_step + 1
       # To prevent warnings casued by race condition, control edge was added
       # More info: https://github.com/google-research/bert/issues/1050
       with tf.control_dependencies([global_step.assign(new_global_step)]):
         train_op = optimizer.apply_gradients(
             zip(grads, tvars), global_step=global_step)
       return train_op
```

Added horovod support:
* Added proper import `from demo.horovod_helpers import hvd, horovod_enabled`
* Added:
```python
      def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
        ...
        if horovod_enabled():
          optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True)
 ```
 Changed:
 ```python
      def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
        ...
        grads = tf.gradients(loss, tvars)
```
to:
```python
      def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
        ...
        grads_and_vars=optimizer.compute_gradients(loss, tvars)
        grads = [grad for grad,var in grads_and_vars]
        tvars = [var for grad,var in grads_and_vars]
```
Added:
```python
      class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
        ...
        def compute_gradients(self, loss, var_list=None,
                              gate_gradients=tf.compat.v1.train.Optimizer.GATE_OP,
                              aggregation_method=None,
                              colocate_gradients_with_ops=False,
                              grad_loss=None):
          assert gate_gradients == tf.compat.v1.train.Optimizer.GATE_OP
          assert aggregation_method is None
          assert not colocate_gradients_with_ops
          assert grad_loss is None
          grads = tf.gradients(ys=loss, xs=var_list)
          grads_and_vars = list(zip(grads, var_list))
          return grads_and_vars
```
Converted to TF2

Changed:
```python
    class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
      ...
      def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        ...
        update = next_m / (tf.sqrt(next_v) + self.epsilon)
```
to:
```python
    class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
      ...
      def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        ...
        update = next_m * tf.math.rsqrt(next_v + self.epsilon*self.epsilon)
```
Due to performance reasons

## modeling.py
Converted to TF2, specifically:

Changed:
```python
       def layer_norm(input_tensor, name=None):
         """Run layer normalization on the last dimension of the tensor."""
         return tf.contrib.layers.layer_norm(
             inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
```
to:
```python
       def layer_norm(input_tensor):
         """Run layer normalization on the last dimension of the tensor."""
         return tf.keras.layers.LayerNormalization(axis=-1, name='LayerNorm')(input_tensor)
```         
In get_assignment_map_from_checkpoint function changed:
```python
      def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
          ...
          assignment_map[name] = name
          ...
```
to
```python
      from tensorflow.python.ops.resource_variable_ops import is_resource_variable
      ...
      def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
          ...
          tvar = name_to_variable[name]
          assert is_resource_variable(tvar)
          assignment_map[name] = tvar
          ...
```
to ensure proper variable initialization after transition to TF2.

## run_squad.py

* Added flags --cpu_only and --do_eval and --use_horovod
* Some minor code refactoring:
   * Pass version_2_with_negative as argument to functions instead of using global FLAGS.version_2_with_negative
   * Moved flags initialization to init_squad_flags function
   * Added `sys.std{out,err}.flush()` at the end of the script
   * Changed import statements to fit our repo directory structure
* Added loading of Habana library
* Added horovod support with our custom hvd functions wrappers (`hvd_XXX` instead of `hvd.XXX`)
   * Initialize hvd
   * Adjust train_batch_size:
   ```python
      train_batch_size = FLAGS.train_batch_size
      if horovod_enabled():
        train_batch_size = train_batch_size * hvd.size()
   ```
   * Adjust num_train_steps:
   ```python
      if horovod_enabled():
          num_train_steps = num_train_steps // hvd.size()
          num_warmup_steps = num_warmup_steps // hvd.size()
   ```
   * Added `hvd.BroadcastGlobalVariablesHook(0)` to estimator
   * Added dataset sharding
   ```python
      def file_based_input_fn_builder(input_file, seq_length, is_training,
                                      drop_remainder):
        ...
        def input_fn(params):
          ...
          if is_training:
            if horovod_enabled():
              d = d.shard(hvd.local_size(), hvd.local_rank())
            ...
   ```
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
   * Changed:
   ```python
      def main(_):
        ...
        if FLAGS.do_train:
          ...
          train_writer = FeatureWriter(
              filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
              ...)
          convert_examples_to_features(
              examples=train_examples,
              ...)
          ...
          train_input_fn = input_fn_builder(
              input_file=train_writer.filename,
          ...)
   ```
   to
   ```python
      def main(_):
        ...
        if FLAGS.do_train:
          ...
          train_writer = FeatureWriter(
              filename=per_worker_filenames[hvd.local_rank() if horovod_enabled() else worker_id],
              ...)
          convert_examples_to_features(
              examples=train_examples[start_index:end_index],
              ...)
          ...
          train_input_fn = input_fn_builder(
              input_file=per_worker_filenames,
          ...)
   ```
* Changed max checkpoint number to 1 (`keep_checkpoint_max=`) to avoid OOM disk usage in multinode
* Added ScopedAllocator support by changing:
  ```python
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=model_dir,
    keep_checkpoint_max=1,
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
* Added flags --use_horovod and --dropout_before_logits (which is passed to tf.nn.dropout)
* Some minor code refatoring:
   * Changed import statements to fit our repo directory structure
* Added loading of Habana library
* Added horovod support, similarly as in run_squad.py except for MultiHLS support
* Added ScopedAllocator support, as in run_squad.py
* Added `SynapseLoggerHook`, as in run_squad.py
* Added``PerfLoggingHook`