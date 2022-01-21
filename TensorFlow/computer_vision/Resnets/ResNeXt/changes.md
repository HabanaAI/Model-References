
### The following are the changes specific to Gaudi that were made to the original scripts:

### imagenet\_main.py

1. Added Habana HPU support
2. Added Horovod support for multinode
3. Added mini_imagenet support
4. Changed the signature of input_fn with new parameters added, and num_parallel_batches removed
5. Changed parallel dataset deserialization to use dataset.interleave instead of dataset.apply
6. Added Resnext support
7. Added parameter to ImagenetModel::__init__ for selecting between resnet and resnext
8. Redefined learning_rate_fn to use warmup_epochs and use_cosine_lr from params
9. Added flags to specify the weight_decay and momentum
10. Added flag to enable horovod support
11. Added calls to tf.compat.v1.disable_eager_execution() and tf.compat.v1.enable_resource_variables() in main code section

### imagenet\_preprocessing.py

1. Changed the image decode, crop and flip function to take a seed to propagate to tf.image.sample_distorted_bounding_box
2. Changed the use of tf.expand_dims to tf.broadcast_to for better performance

### resnet\_model.py

1. Added tf.bfloat16 to CASTABLE_TYPES
2. Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2
3. Deleted the call to pad the input along the spatial dimensions independently of input size for stride greater than 1
4. Added functionality for strided 2-D convolution with groups and explicit padding
5. Added functionality for a single block for ResNext, with a bottleneck
6. Added Resnext support

### resnet\_run\_loop.py

1. Changes for enabling Horovod, e.g. data sharding in multi-node, usage of Horovod's TF DistributedOptimizer, etc.
2. Added options to enable the use of tf.data's 'experimental_slack' and 'experimental.prefetch_to_device' options during the input processing
3. Added support for specific size thread pool for tf.data operations
4. TensorFlow v2 support: Changed dataset.apply(tf.contrib.data.map_and_batch(..)) to dataset.map(..., num_parallel_calls, ...) followed by dataset.batch()
5. Changed calls to tf.<api> to tf.compat.v1.<api> for backward compatibility when running training scripts on TensorFlow v2
6. Other TF v2 replacements for tf.contrib usages
7. Redefined learning_rate_with_decay to use warmup_epochs and use_cosine_lr
8. Added functionality for label smoothing
9. Commented out writing images to summary, for performance reasons
10. Added check for non-tf.bfloat16 input images having the same data type as the dtype that training is run with
11. Added functionality to define the cross-entropy function depending on whether label smoothing is enabled
12. Added support for loss scaling of gradients
13. Added flag for experimental_preloading that invokes the HabanaEstimator, besides other optimizations such as tf.data.experimental.prefetch_to_device
14. Added 'TF_DISABLE_SCOPED_ALLOCATOR' environment variable flag to disable Scoped Allocator Optimization (enabled by default) for Horovod runs
15. Added a flag to configure the save_checkpoint_steps
16. If the flag "use_train_and_evaluate" is set, or in multi-worker training scenarios, there is a one-shot call to tf.estimator.train_and_evaluate
17. resnet_main() returns a dictionary with keys 'eval_results' and 'train_hooks'
18. Added flags in 'define_resnet_flags' for flags_core.define_base, flags_core.define_performance, flags_core.define_distribution, flags_core.define_experimental, and many others (please refer to this function for all the flags that are available)
19. Changed order of ops creating summaries to log them in TensorBoard with proper name. Added saving HParams to TensorBoard and exposed a flag for specifying frequency of summary updates.
20. Changed a name of directory, in which workers are saving logs and checkpoints, from "rank_N" to "worker_N".

