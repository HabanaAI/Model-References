# BERT Pretraining changes

Originally, scripts were taken from [NVidia Github](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT), commit hash [3337f72cf](https://github.com/NVIDIA/DeepLearningExamples/tree/2729732c31939bc52bd04f9ea3c12334af119f47/TensorFlow/LanguageModeling/BERT)

Files used:
* fused_layer_norm.py
* gpu_environment.py
* modeling.py
* optimization.py
* run_pretraining.py
* utils/
  * dllogger_class.py
  * utils.py

All of above files were converted to TF2 by using [tf_upgrade_v2](https://www.tensorflow.org/guide/upgrade?hl=en) tool. Additionally, some other changes were committed for specific files. Also, in these files, horovod usage was changed to use horovod functions wrappers and global `hvd` object instead of passing `hvd` as an argument:

* Added

`from TensorFlow.common.horovod_helpers import hvd, hvd_init, hvd_size, hvd_rank, horovod_enabled, comm_local_rank`

* Changed usage of ``hvd.XXXX`` to ``hvd_XXXX``
* Changed ``hvd is not None`` to ``horovod_enabled()``
* Changed ``FLAGS.horovod and hvd.size() > 1`` to ``horovod_enabled()``

## modeling.py
* In get_assignment_map_from_checkpoint function changed to:
  ```python 
   from tensorflow.python.ops.resource_variable_ops import is_resource_variable ... 
   def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
   ..
     tvar = name_to_variable[name] assert is_resource_variable(tvar) assignment_map[name] = tvar ..
  ```
to ensure proper variable initialization after transition to TF2.
  
Changed scope and layer norm function in: 
```python
def layer_norm(input_tensor, name=None):
"""Run layer normalization on the last dimension of the tensor.""" 
if input_tensor.dtype == tf.float16:
try:
from fused_layer_norm import fused_layer_norm return fused_layer_norm(
   inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name, use_fused_batch_norm=True)

except ImportError:
return tf.contrib.layers.layer_norm(
 inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
 else:
 return tf.contrib.layers.layer_norm(
 inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
```
to
 
```python
def layer_norm(input_tensor):
  """Run layer normalization on the last dimension of the tensor."""
  if input_tensor.dtype == tf.float16:
   try:
    from fused_layer_norm import fused_layer_norm
     return fused_layer_norm(
       inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope='LayerNorm',
       use_fused_batch_norm=True)
   except ImportError:
     return tf.keras.layers.LayerNormalization(axis=-1, name='LayerNorm')(input_tensor)
else:
 return tf.keras.layers.LayerNormalization(axis=-1, name='LayerNorm')(input_tensor)
```
because argument ``name`` was never used and ``tf.contrib.layers.layer_norm`` is no longer available in TF2.
   
## optimization.py

* Used custom ``horovod_enabled()`` imported with ``from TensorFlow.common.horovod_helpers import hvd, horovod_enabled``, instead of ``hvd is None``
* Removed some fp16-related parts, mainly ``compression=ompression.fp16 if use_fp16 or manual_fp16 else Compression.none``

Changed
```python 
def update(accum_vars):

 if allreduce_post_accumulation and hvd is not None:
  accum_vars = [hvd.allreduce(tf.convert_to_tensor(accum_var), compression=Compression.fp16 if use_fp16 or manual_fp16 else Compression.none) if isinstance(accum_var, tf.IndexedSlices)
   else hvd.allreduce(accum_var, compression=Compression.fp16 if use_fp16 or manual_fp16 
   else Compression.none) for accum_var in accum_vars]
  
  return optimizer.apply_gradients(list(zip(accum_vars, tvars)), global_step=global_step)

 update_step = tf.identity(tf.cast(tf.math.equal(local_step % num_accumulation_steps, 0), dtype=tf.bool), name="update_step")
 update_op = tf.cond(update_step,

 lambda: update(accum_vars), lambda: tf.no_op())

new_global_step = tf.cond(tf.math.logical_and(update_step,
 tf.cast(hvd.allreduce(tf.cast(batch_finite, tf.int32)), tf.bool) if hvd is not None else batch_finite),lambda: global_step+1,lambda: global_step)
      new_global_step = tf.identity(new_global_step, name='step_update')
      train_op = tf.group(update_op, [global_step.assign(new_global_step)])
```
to
```python
def allreduce_of_batch_finite_required():
# In case of bf16 and fp32 batch finite is tf.constant(True, dtype=tf.bool) return horovod_enabled() and manual_fp16 and use_fp16
      
new_global_step = tf.cond(pred=tf.math.logical_and(update_step,
 tf.cast(hvd.allreduce(tf.cast(batch_finite, tf.int32)), tf.bool) if allreduce_of_batch_finite_required() else batch_finite),true_fn=lambda: global_step+1,false_fn=lambda: global_step)

new_global_step = tf.identity(new_global_step, name='step_update')
      def update(accum_vars):
        with tf.control_dependencies([global_step.assign(new_global_step)]):
          if allreduce_post_accumulation and horovod_enabled():
              accum_vars = [hvd.allreduce(tf.convert_to_tensor(value=accum_var)) if isinstance(accum_var, tf.IndexedSlices)
                            else hvd.allreduce(accum_var) for accum_var in accum_vars]
          return optimizer.apply_gradients(list(zip(accum_vars, tvars)), global_step=global_step)
      train_op = tf.cond(pred=update_step,
                          true_fn=lambda: update(accum_vars), false_fn=lambda: tf.no_op())
```

The major part here is to add control edge before updating global_step - this is to prevent situation, where global_step doesn't get increased in first iteration to 1 before following computations, which can guide to division by beta1_correction=0 otherwise:
  
```python
class LAMBOptimizer(tf.compat.v1.train.Optimizer):
 ..def apply_gradients(self, grads_and_vars, global_step, name=None,manual_fp16=False):
 ..steps = tf.cast(global_step, tf.float32)
        for (grad, param) in grads_and_vars:
          ..
          beta1_correction = (1 - self.beta_1 ** steps)
          beta2_correction = (1 - self.beta_2 ** steps)
          next_m_unbiased = next_m / beta1_correction
          next_v_unbiased = next_v / beta2_correction
  ```

Minor part is that we optimized some unnecessary computations meant to be executed when fp16 is enabled by introducing ``allreduce_of_batch_finite_required``.

Changed
`grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None] grads, tvars = list(zip(*grads_and_vars))`
to
```python
 if horovod_enabled():
   grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None] grads, tvars = list(zip(*grads_and_vars))
 else:
   grads = tf.gradients(ys=loss, xs=tvars)
```
Changed
```python
 train_op = optimizer.apply_gradients(
  list(zip(clipped_grads, tvars)), global_step=global_step)
 new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
 new_global_step = tf.identity(new_global_step, name='step_update')
 train_op = tf.group(train_op, [global_step.assign(new_global_step)])
```
to
```python
 new_global_step = tf.cond(pred=all_are_finite, true_fn=lambda: global_step + 1, false_fn=lambda: global_step)
 new_global_step = tf.identity(new_global_step, name='step_update')
   with tf.control_dependencies([global_step.assign(new_global_step)]):
     train_op = optimizer.apply_gradients(
     list(zip(clipped_grads, tvars)), global_step=global_step)
```
due to the same division by 0 prevention as above.
   
## run_pretraining.py
Introduced
```python
    class _LogSessionRunHook(tf.estimator.SessionRunHook):
      ...
      def before_run(self, run_context):
        if horovod_enabled() and hvd_rank() != 0:
          return
        ...
      def after_run(self, run_context, run_values):
        if horovod_enabled() and hvd_rank() != 0:
          return
```
and changed:
```python
    def main(_):
    ...if (not FLAGS.horovod or hvd.rank() == 0):
        global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps if not FLAGS.horovod else FLAGS.train_batch_size * FLAGS.num_accumulation_steps * hvd.size()
        training_hooks.append(_LogSessionRunHook(global_batch_size, FLAGS.num_accumulation_steps, dllogging, FLAGS.display_loss_steps, FLAGS.save_checkpoints_steps,FLAGS.report_loss))
```
to:
```python
    def main(_):
      ...
      global_batch_size = FLAGS.train_batch_size * FLAGS.num_accumulation_steps if not FLAGS.horovod else FLAGS.train_batch_size * FLAGS.num_accumulation_steps * hvd.size()
      training_hooks.append(_LogSessionRunHook(global_batch_size, FLAGS.num_accumulation_steps, dllogging, FLAGS.display_loss_steps, FLAGS.save_checkpoints_steps, FLAGS.report_loss))
   to prevent stalled tensors caused by differences in graphs between node with rank=0 and the rest of nodes.
```

Changed behavior not to log performance if num_train_steps < 6 to prevent ZeroDivisionError
Changed:
```python
    def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps,
                         use_one_hot_embeddings, hvd=None):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        ...
        if init_checkpoint and (hvd is None or hvd.rank() == 0):
          print("Loading checkpoint", init_checkpoint)
          ...
```
to
```python
    def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps,
                         use_one_hot_embeddings):
      """Returns `model_fn` closure for TPUEstimator."""
      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        ...
        if init_checkpoint:
          ...
   
```
Added
```python
    def main(_):
      ...
      # In multi-node scenario, on each of HLSes there must be a checkpoint directly in the output_dir (read by Phase 2).
      # There may be only one worker with comm_local_rank() == 0 on each machine and this worker will put its checkpoints there.
      # All other workers use sub-directories to keep checkpoints.
      if horovod_enabled() and comm_local_rank() != 0:
        FLAGS.output_dir = os.path.join(FLAGS.output_dir, str(hvd_rank()))
```

Changed:
```python
    def main(_):
      ...
      if FLAGS.horovod and len(input_files) < hvd.size():
          raise ValueError("Input Files must be sharded")
```   
to:
```python
    def main(_):
      ...
      if FLAGS.horovod and len(input_files) < hvd.size():
          tf.compat.v1.logging.warning("Input files count lower then expected. Using single file for OVERFIT test.")
          input_files = [input_files[0] for i in range(hvd.size())]
```
Added Scoped Allocator. 

Changed:
```python
    def main(_):
      ...
      config = tf.compat.v1.ConfigProto()
```   
to
```python
    def main(_):
      ...
      if condition_env_var('TF_DISABLE_SCOPED_ALLOCATOR', default=False):
        session_config = tf.compat.v1.ConfigProto()
      else:
        from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=import-error
        session_config = tf.compat.v1.ConfigProto()
        session_config.graph_options.rewrite_options.scoped_allocator_optimization = rewriter_config_pb2.RewriterConfig.ON
        enable_op = session_config.graph_options.rewrite_options.scoped_allocator_opts.enable_op
        del enable_op[:]
        enable_op.append("HorovodAllreduce")
```
Changed:
```python
    def main(_):
      ...
      run_config = tf.estimator.RunConfig(
          ...
          save_checkpoints_steps=FLAGS.save_checkpoints_steps if not FLAGS.horovod or hvd.rank() == 0 else None,
          save_summary_steps=FLAGS.save_checkpoints_steps if not FLAGS.horovod or hvd.rank() == 0 else None,
          ...)
 ```  
 to:
 ```python
    def main(_):
      ...
      run_config = tf.estimator.RunConfig(
          ...
          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
          save_summary_steps=FLAGS.save_checkpoints_steps if not FLAGS.horovod else None,
          ...)
``` 
to prevent stalled ranks.
Changed behavior to log iteration info every 1 step, regardless of --report_loss flag.
Added `load_habana_module()`
Added WeightAllreducingHook:
```python
    class TrainableVarsAllreducingHookPreOpt(tf.compat.v1.estimator.SessionRunHook):
      def __init__(self, num_accumulation_steps=1):
        super(TrainableVarsAllreducingHookPreOpt, self).__init__()
        # Modify this collection in order to allreduce other set of variables
        trainable_vars = tf.compat.v1.trainable_variables()
        allreduced_trainable_var_ops = [ v.assign(hvd.allreduce(v)) for v in trainable_vars]
        self.allreduce_trainable_vars_op = tf.group(*allreduced_trainable_var_ops)
        self.num_accumulation_steps = num_accumulation_steps
        self.current_iteration = 1
      def before_run(self, run_context):
        if self.current_iteration % self.num_accumulation_steps == 0:
          return tf.compat.v1.train.SessionRunArgs(self.allreduce_trainable_vars_op)
      def after_run(self, run_context, run_values):
        self.current_iteration += 1
        trainable_vars_allreduce_result = run_values.results
```
and:
```python
    if os.environ.get("FORCE_WEIGHT_SYNC", "False").lower() in ["true", "1"]:
        # Use this hook to allreduce trainable variables before the optimizer run
        training_hooks.append(TrainableVarsAllreducingHookPreOpt(FLAGS.num_accumulation_steps))
```
Changed:
```python
    def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                            label_ids, label_weights):
      """Get loss and log probs for the masked LM."""
      ...
      log_probs = tf.nn.log_softmax(logits, axis=-1)
```
to:
```python
    def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                            label_ids, label_weights):
      """Get loss and log probs for the masked LM."""
      ...
      log_probs = tf.nn.log_softmax(logits - tf.reduce_max(logits, keepdims=True, axis=-1), axis=-1)
```
and:
```python
    def get_next_sentence_output(bert_config, input_tensor, labels):
       """Get loss and log probs for the next sentence prediction."""
      ...
      log_probs = tf.nn.log_softmax(logits, axis=-1)
```
to:
```python
    def get_next_sentence_output(bert_config, input_tensor, labels):
      """Get loss and log probs for the next sentence prediction."""
      ...
      log_probs = tf.nn.log_softmax(logits - tf.reduce_max(logits, keepdims=True, axis=-1), axis=-1)
```
Change makes log_softmax more stable.
