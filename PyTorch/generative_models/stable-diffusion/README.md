# Stable Diffusion for PyTorch

This directory provides scripts to train Stable Diffusion Model which is based on latent text-to-image diffusion model and is tested and maintained by Intel® Gaudi®.
For more information on training and inference of deep learning models using Intel Gaudi AI accelerator, refer to [developer.habana.ai](https://developer.habana.ai/resources/). Before you get started, make sure to review the [Supported Configuration](#supported-configuration).

  - [Model-References](../../../README.md)
  - [Model Overview](#model-overview)
  - [Setup](#setup)
  - [Training and Examples](#training-and-examples)
  - [Supported Configuration](#supported-configuration)
  - [Known Issues](#known-issues)
  - [Changelog](#changelog)


## Model Overview
This implementation is based on the following paper - [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752).
This model uses a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a HPU.

### How to Use
Users acknowledge and understand that the models referenced by Habana are mere examples for models that can be run on Gaudi.
Users bear sole liability and responsibility to follow and comply with any third party licenses pertaining to such models,
and Habana Labs disclaims and will bear no any warranty or liability with respect to users' use or compliance with such third party licenses.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi.  

### Clone Intel Gaudi Model-References
In the docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.
```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/generative_models/stable-diffusion
```

### Install Model Requirements
1. In the docker container, go to the model directory:
```bash
cd Model-References/PyTorch/generative_models/stable-diffusion
```

2. Install the required packages using pip:
```bash
pip install -r requirements.txt
export PYTHONPATH=$MODEL_REF_PYTORCH_PATH/generative_models/stable-diffusion/src/taming-transformers:$PYTHONPATH
```
Note: Ensure `MODEL_REF_PYTORCH_PATH` is set to Model-References/PyTorch before setting PYTHONPATH

### Model Checkpoint

Download the model checkpoint for `first_stage_config` by going to https://ommer-lab.com/files/latent-diffusion/
location, save kl-f8.zip to your local folder and unzip it. Make sure to point the `ckpt_path` in `hpu_config_web_dataset.yaml`
file to the full path of the checkpoint.

Users acknowledge and understand that by downloading the checkpoint referenced herein they will be required to comply
with third party licenses and rights pertaining to the checkpoint, and users will be solely liable and responsible
for complying with any applicable licenses. Habana Labs disclaims any warranty or liability with respect to users' use
or compliance with such third party licenses.

### Dataset Preparation

Users acknowledge and understand that by downloading the dataset referenced herein they will be required to comply
with third party licenses and usage rights to the data, and users will be solely liable and responsible for
complying with any applicable licenses. Habana Labs disclaims any warranty or liability with respect to users' use
or compliance with such third party licenses.

Training on stable-diffusion is performed using laion2B-en dataset: https://huggingface.co/datasets/laion/laion2B-en

1. To download Laion-2B-en dataset, run the following. The below method downloads the dataset locally when not using S3 bucket.
```bash
pip install img2dataset
mkdir laion2B-en && cd laion2B-en
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-en/resolve/main/part-$i-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet; done
cd ..
```
2. Create a download.py file and add following:

```bash
from img2dataset import download
url_list ="laion2B-en/"
output_dir = "laion-data/laion2B-data"

if __name__ == '__main__':
    download(
        processes_count=16,
        thread_count=32,
        url_list = url_list,
        image_size=384,
        resize_only_if_bigger=True,
        resize_mode="keep_ratio",
        skip_reencode=True,
        output_folder=output_dir,
        output_format="webdataset",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        number_sample_per_shard=10000,
        distributor="multiprocessing",
        save_additional_columns=["NSFW","similarity","LICENSE"],
        oom_shard_count=6,
    )
```
In addition,
"processes_count" and "thread_count" are tunable parameters based on the host machine where the dataset is downloaded.
"image_size" can be modified as per the requirement of the training checkpoint.

3. Download the dataset:
```bash
python download.py
```
Note:
 * The dataset used for training is unavailable due to issues described in https://laion.ai/notes/laion-maintanence/.
 * Generic data preparation can be found https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md

## Training and Examples
### Single Card and Multi-Card Training Examples
**Run training on 1 HPU:**
* HPU, Lazy mode with FP32 precision:
```
python main.py --base hpu_config_web_dataset.yaml --train --scale_lr False --seed 0 --hpus 1 --batch_size 4 --use_lazy_mode True --no-test True --max_epochs 10 --limit_train_batches 1000 --limit_val_batches 0  --hpu_graph True --ckpt_path="/software/lfs/data/pytorch/stable-diffusion/checkpoint/model.ckpt" --dataset_path="/software/lfs/data/pytorch/stable-diffusion/laion2B-data/"
```
* HPU, Lazy mode with BF16 mixed precision:
```
python main.py --base hpu_config_web_dataset.yaml --train --scale_lr False --seed 0 --hpus 1 --batch_size 8 --use_lazy_mode True --autocast --no-test True --max_epochs 10 --limit_train_batches 1000 --limit_val_batches 0  --hpu_graph True --ckpt_path="/software/lfs/data/pytorch/stable-diffusion/checkpoint/model.ckpt" --dataset_path="/software/lfs/data/pytorch/stable-diffusion/laion2B-data/"
```
**Run training on 8 HPUs:**
* 8 HPUs, Lazy mode with FP32 precision:
```
python main.py --base hpu_config_web_dataset.yaml --train --scale_lr False --seed 0 --hpus 8 --batch_size 4 --use_lazy_mode True --no-test True --max_epochs 10 --limit_train_batches 1000 --limit_val_batches 0  --hpu_graph True --ckpt_path="/qa/stable_diffusion/models/checkpoint/first_stage_models/kl-f8/model.ckpt" --dataset_path="/software/lfs/data/pytorch/stable-diffusion/laion2B-data/"
```
* 8 HPUs, Lazy mode with BF16 mixed precision:
```
python main.py --base hpu_config_web_dataset.yaml --train --scale_lr False --seed 0 --hpus 8 --batch_size 8 --use_lazy_mode True --autocast --no-test True --max_epochs 10 --limit_train_batches 1000 --limit_val_batches 0  --hpu_graph True --ckpt_path="/software/lfs/data/pytorch/stable-diffusion/checkpoint/model.ckpt" --dataset_path="/mnt/weka/data/pytorch/stable-diffusion/laion2B-data/"
```

**Note**
`--limit_train_batches` specifies the length of the training loop/number of batches to run in each epoch
For first-gen Gaudi, use `--hpu_graph` False to avoid device OOM issue.

### Multi-Server Training Examples
**Run training on 16 HPUs:**
Log in to both worker machines in separate shells and set the below environment variables:
Worker-0
```
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
export NODE_RANK=0

python main.py --base hpu_config_web_dataset.yaml --train --scale_lr False --seed 0 --hpus 8 --batch_size 8 --use_lazy_mode True --autocast --no-test True --max_epochs 10 --limit_train_batches 1000 --limit_val_batches 0  --hpu_graph True --ckpt_path="/software/lfs/data/pytorch/stable-diffusion/checkpoint/model.ckpt" --dataset_path="/software/lfs/data/pytorch/stable-diffusion/laion2B-data/"  --num_nodes=2
```
Worker-1
```
export MASTER_ADDR="10.10.100.101"
export MASTER_PORT="12345"
export NODE_RANK=1

python main.py --base hpu_config_web_dataset.yaml --train --scale_lr False --seed 0 --hpus 8 --batch_size 8 --use_lazy_mode True --autocast --no-test True --max_epochs 10 --limit_train_batches 1000 --limit_val_batches 0  --hpu_graph True --ckpt_path="/software/lfs/data/pytorch/stable-diffusion/checkpoint/model.ckpt" --dataset_path="/software/lfs/data/pytorch/stable-diffusion/laion2B-data/"  --num_nodes=2
```

## Supported Configuration
| Validated on  | Intel Gaudi Software Version | PyTorch Version | PyTorch Lightning Version| Mode |
|---------|-------------------|-----------------|--------------|-----------------------------------|
| Gaudi   | 1.11.0             | 2.0.1          | 2.0.6 | Training |
| Gaudi 2 | 1.17.0             | 2.3.1          | 2.3.3 | Training |

## Known Issues
* Model was trained using "laion2B-en" dataset for limited number of steps with `batch_size: 8` and `accumulate_grad_batches: 16`.
* ImageLogger callbacks as part of training is enabled and tested only with options given `hpu_config_web_dataset.yaml`.
* For first-gen Gaudi, use `--hpu_graph` False to avoid device OOM issue in training.
* Only scripts and configurations mentioned in this README are supported and verified.

## Changelog

### Script Modifications
Major changes done to the original model from [pesser/stable-diffusion](https://github.com/pesser/stable-diffusion/commit/693e713c3e72e72b8e2b97236eb21526217e83ad) repository:
### 1.15.0
* Removed inference flow and all changes related to it.
### 1.13.0
* Added support for lightning version 2.0.9.
* PTL version upgraded to 2.0.9.
* Added support for lightning-habana 1.1.0.
### 1.12.0
* Removed support for HMP.
* PTL version upgraded to 2.0.6.
* Added support for checkpoint resume.
* Eager mode support is deprecated.
### 1.11.0
* Dynamic Shapes will be enabled by default in future releases. It is currently enabled in training script as a temporary solution.
### 1.10.0
* Enabled PyTorch autocast on Gaudi.
* Tensorboard/Wandb logger issue in training step resolved (disabled by default) along with new command options introduced for logger.
* Upgraded PTL version from 1.9.4 to 2.0.0, and modified the script accordingly.
### 1.9.0
* DDP Paramters were tuned for multicard run.
* Made accumulate_grad_batches 16 as the default instead of 1.
* Moved to web dataset instead of local dataset. Added required changes in dataset reading interface.
* In BasicTransformerBlock, init function checkpoint is disabled by default.
* In AttentionBlock, use_checkpoint value is taken from config instead of always true.
* First stage model and FrozenClip Models are wrapped using HPU Graphs.
* Diffusion model is wrapped using HPU Graphs. Added `--hpu_graph` option to enbale/disable this optimization.
* Additional arguments were added to modify the checkpoint and dataset path.
* PTL version changed from 1.7.7 to 1.9.4 with required modifications in scripts.
* Added `--image_logger` option to enable/disable image logger during the training. `--image_logger` is by default enabled and can be set to False to disable.
* Training config file(yaml) added with image logger options.
* Added inference flow.
* Changed default config and ckpt in scripts/txt2img.py.
* Changed logic in ldm/models/diffusion/ddim.py in order to avoid graph recompilations.
* Added interactive mode for demonstrative purposes.
* Fixed python insecurity in requirements.txt

### 1.8.0
* Changed README file content.
* environment.yaml is replaced from CompVis's stable-diffusion.
* Changed the basic implementation of dataset for reading files from directory in img2dataset's format(ldm/data/base.py).
* Changed default precision from autocast to full.
* PTL version changed from 1.4.2 to 1.7.7 with required modification in scripts and torchmetrics from 0.6 to 0.10.3.
* ImageLogger callback disabled while running on HPU.
* HPU support for single and multi card(using HPUParallelStrategy) is enabled.Also tuned parameter values for DDP.
* HMP support added for mixed precision training with required ops in fp32 and bf16.
* HPU and CPU config files were added.
* Introduced additional mark_steps as per need in the Model.
* Added FusedAdamw optimizer support nd made it enabled by default.
* Introduced the print frequency changes to extract the loss as per the user configured value from `refresh_rate`.
* Added changes in dataloader for multicard.
