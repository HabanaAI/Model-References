# central directory for all models in Model-References

The central directory contains code used by TensorFlow and PyTorch models:

- **habana_model_runner.py** : Main Python training runner, requires these CLI options:
  --framework <tensorflow | pytorch>
  --model <model_name>
  --hb_config <config.yaml>
  Example: python3 /path/to/Model-References/central/habana_model_runner.py --framework tensorflow --model bert --hb_config hb_configs/bert_base_mrpc.yaml

- **script_paths.py** : Dictionary of <model_name> to <Python_training_script_path> for TensorFlow models and for PyTorch models

- **habana_model_runner_utils.py** : Utility functions for use in Python training scripts, e.g. "HabanaEnvVariables" to define a context with specific environment variable settings in which a training script can be run

- **habana_model_yaml_config.py** : Class for parsing and querying a yaml configuration file containing model hyperparameters and training run parameters

- **generate_hcl_config.py** : Function for generating an HCL configuration file at specified location, with given "number of devices per HLS" and given "number of HLS nodes", where HLS nodes' IP addresses are specified in the MULTI_HLS_IPS environment variable. Python training scripts invoke this on a single node or on each remote node.

- **check_dirs.py** : Function for checking the existence of specific files/directories, at specific stages during training. Python training scripts invoke this on a single node or on each remote node.

- **prepare_output_dir.py** : Function for creating a training run output directory at specified path. Python training scripts invoke this on a single node or on each remote node.

- **multi_node_utils.py** : Utilities for running a command as subprocess on local host or on each remote node configured in MULTI_HLS_IPS environment variable for scaleout training, generating MPI hostfile, etc.

- **training_run_config.py** : Class that encapsulates the hardware configuration for scaleout training using mpirun with Horovod so derived classes can expect a fully-configured mpirun command-line for 1-card, 8-cards, or multi-HLS distributed training, and can know whether or not Horovod is enabled
