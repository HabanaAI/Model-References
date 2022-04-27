# central directory for all models in Model-References

The central directory contains code used by TensorFlow and PyTorch models:

- **script_paths.py** : Dictionary of <model_name> to <Python_training_script_path> for TensorFlow models and for PyTorch models

- **check_dirs.py** : Function for checking the existence of specific files/directories, at specific stages during training. Python training scripts invoke this on a single node or on each remote node.

- **prepare_output_dir.py** : Function for creating a training run output directory at specified path. Python training scripts invoke this on a single node or on each remote node.

- **multi_node_utils.py** : Utilities for running a command as subprocess on local host or on each remote node configured in MULTI_HLS_IPS environment variable for scaleout training, generating MPI hostfile, etc.

- **training_run_config.py** : Class that encapsulates the hardware configuration for scaleout training using mpirun with Horovod so derived classes can expect a fully-configured mpirun command-line for 1-card, 8-cards, or multi-HLS distributed training, and can know whether or not Horovod is enabled

