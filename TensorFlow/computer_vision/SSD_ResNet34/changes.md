# SSD-ResNet34 model changes
The SSD ResNet34 model is based on https://github.com/mlperf/training_results_v0.6/ (Google/benchmarks/ssd/implementations/tpu-v3-32-ssd)

## Model Changes
*  Migrated from TF 1.15 to TF 2.2
*  Disabled Eager mode
*  Enabled resource variables
*  TPU-specific topk_mask implementation is replaced with implementation from the reference
*  Removed TPU and GCP related code
*  Added Horovod for distibuted training
*  Added support for HPU (load_habana_modules and Habana Horovod)
*  Used argparse instead of tf.flags
*  Removed mlperf 0.6 logger
*  Added demo_ssd allowing to run multinode training with OpenMPI
*  Added flags: num_steps, log_step_count_steps, save_checkpoints_steps
*  Turned off dataset caching when RAM size is not sufficient
*  Added 'use-fake-data' flag (allows to measure performance without dataloader overhead)
*  Added support for TF profiling
*  Used dataset.interleave instead of tf.data.experimental.parallel_interleave in dataloader.py
*  Name scopes 'concat_cls_outputs' and 'concat_box_outputs' added to concat_outputs
*  Added inference mode
*  Fixed issues with COCO2017 dataset - provided script for generating correct dataset, 117266 training examples
*  Boxes and classes are transposed in dataloder not in model to improve performance
*  Removed weight_decay loss from total loss calculation (fwd only)
*  Updated input normalization formula to use multiplication by reciprocal instead of division
*  Added logging hook that calculates IPS and total training time
*  Introduced custom softmax_cross_entropy_mme loss function that better utilizes HPU hardware (by implementing reduce_sum through conv2d which is computed on MME in parallel with other TPC operations and transposing tensors for reduce_max)
*  Added support for distributed batch normalization
