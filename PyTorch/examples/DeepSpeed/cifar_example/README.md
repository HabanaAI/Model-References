# Getting Started with DeepSpeed
This guide provides simple steps for preparing a DeepSpeed model to run on Intel® Gaudi® AI accelerator. Make sure to install the DeepSpeed package provided by Intel Gaudi. Installing public DeepSpeed packages is not supported.

To set up the environment, refer to the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html#gaudi-installation-guide). The supported DeepSpeed versions are listed in the [Support Matrix](https://docs.habana.ai/en/latest/Support_Matrix/Support_Matrix.html#support-matrix).

## Start Training a DeepSpeed Model on Gaudi
Steps to run the model are listed below, for the detailed setup instructions, please see the [Getting Started With DeepSpeed](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Getting_Started_with_DeepSpeed/Getting_Started_with_DeepSpeed.html) documentation:
1. Run Intel Gaudi PyTorch Docker image.
2. Install Intel Gaudi's DeepSpeed fork.
3. Clone the Model-References GitHub repository inside the container that you have just started.
4. Move to this examples folder and install the associated requirements and PYTHONPATH env variable.
5. Execute the `run_ds_habanax8.sh` script (if you are running on only 1 HPU, please modify the script to set `--num_gpus=1`):
    ```bash
    deepspeed --num_nodes=1 --num_gpus=8 cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json --use_hpu
    ```
At the end of the training run, you will see a command indicating that the `Process 3226 exits successfully`.

The user can review the following files for the specific DeepSpeed or Intel Gaudi-related information
* `ds_config.json` file for the DeepSpeed specific changes and steps in the model.
* `run_ds_habanax8.sh` for the specific DeepSpeed run command, specifically the `--use_hpu` flag to ensure that the model is running on the Gaudi.
* `cifar10_deepspeed.py` for the full model script. The Intel Gaudi-specific changes have been added in the comments.
