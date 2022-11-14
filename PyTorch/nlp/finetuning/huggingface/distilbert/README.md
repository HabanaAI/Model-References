# DistilBert for PyTorch

This folder contains Squad fine-tune DistilBERT model on Habana Gaudi device to achieve state-of-the-art accuracy. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information about training deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Fine-tuning](#fine-tuning)
* [Training Examples ](#training-examples)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Citation](#citation)

## Model Overview

Distil* is a class of compressed models that started with DistilBERT. DistilBERT stands for Distilled-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving 97% of BERT's performances as measured on the GLUE language understanding benchmark. DistilBERT is trained using knowledge distillation, a technique to compress a large model called the teacher into a smaller model called the student. By distillating Bert, we obtain a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilBERT is thus an interesting option to put large-scaled trained Transformer model into production.

This repository is a copy of the distillation scripts on the original Huggingface repository (<https://github.com/huggingface/transformers.git>) at transformers/examples/research_projects/distillation/, commit 3b1f5caff26c08dfb74a76de1163f4becde9e828 from June 11, 2021.

## Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
This guide will walk you through the process of setting up your system to run
the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. (Run the hl-smi utility to determine the SynapseAI version.)
```bash
 git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

Then, install the transformers module:
```bash
pip install Model-References/PyTorch/nlp/finetuning/huggingface/bert/transformers/.
```

Go to PyTorch distilbert directory:
```bash
cd Model-References/PyTorch/nlp/finetuning/huggingface/distilbert
```

### Install Model Requirements
In the docker container, install the required packages using pip:
```bash
$PYTHON -m pip install -r requirements.txt
```

### SQuADv1.1 Dataset Preparation

Public datasets are available on the datasets hub: <https://github.com/huggingface/datasets>.

## Fine-tuning

The current script only has teacher_type bert case fully tested.

* teacher_type Bert fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in eager mode.
* teacher_type Bert fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in lazy mode.

The demo script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.


* Located in: `Model-References/PyTorch/nlp/distilbert/`
* Suited for task. 
  * **squad:** Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
* Uses optimizer: **AdamW** (“ADAM with Weight Decay Regularization”).
* Light-weight
* Datasets for SQuAD needs to be downloaded to the docker container.

## Training Examples

### Single Card and Multi-Card Training Examples
**Run training on 1 HPU:**

- Run DistilBert using FP32 data type, teacher_type bert, Eager mode:

   ```

   $PYTHON run_squad_w_distillation.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --teacher_type=bert \
      --teacher_name_or_path=bert-base-uncased --config_name=./training_configs/distilbert-base-uncased.json --train_file=/data/pytorch/transformers/Squad/train-v1.1.json \
      --predict_file=/data/pytorch/transformers/Squad/dev-v1.1.json --do_eval --do_train --do_lower_case --output_dir=/tmp/distbert/tmp_train --overwrite_output_dir \
      --hpu --optimizer=FusedAdamW --hmp_opt_level=O1 --save_steps=50 --num_train_epochs=3.0 --world_size=1

   ```

- Run DistilBert using BF16 data type, teacher_type bert, Eager mode:

   ```

   $PYTHON run_squad_w_distillation.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --teacher_type=bert \
      --teacher_name_or_path=bert-base-uncased --config_name=./training_configs/distilbert-base-uncased.json --train_file=/data/pytorch/transformers/Squad/train-v1.1.json \
      --predict_file=/data/pytorch/transformers/Squad/dev-v1.1.json --do_eval --do_train --do_lower_case --output_dir=/tmp/distbert/tmp_train --overwrite_output_dir \
      --hpu --optimizer=FusedAdamW --hmp --hmp_bf16=./ops_bf16_distilbert_pt.txt --hmp_fp32=./ops_fp32_distilbert_pt.txt --hmp_opt_level=O1 --save_steps=50 \
      --num_train_epochs=3.0 --world_size=1

   ```

- Run DistilBert using FP32 data type, teacher_type bert, Lazy mode:
  ```

   $PYTHON run_squad_w_distillation.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --teacher_type=gibert \
      --teacher_name_or_path=bert-base-uncased --config_name=./training_configs/distilbert-base-uncased.json --train_file=/data/pytorch/transformers/Squad/train-v1.1.json \
      --predict_file=/data/pytorch/transformers/Squad/dev-v1.1.json --do_eval --do_train --do_lower_case --output_dir=/tmp/distbert/tmp_train --overwrite_output_dir \
      --hpu --optimizer=FusedAdamW --hmp_opt_level=O1 --use_lazy_mode --save_steps=1000 --num_train_epochs=3.0 --world_size=1

   ```
- Run DistilBert using BF16 data type, teacher_type bert, Lazy mode:

   ```

   $PYTHON run_squad_w_distillation.py --model_type=distilbert --model_name_or_path=distilbert-base-uncased --teacher_type=bert \
      --teacher_name_or_path=bert-base-uncased --config_name=./training_configs/distilbert-base-uncased.json --train_file=/data/pytorch/transformers/Squad/train-v1.1.json \
      --predict_file=/data/pytorch/transformers/Squad/dev-v1.1.json --do_eval --do_train --do_lower_case --output_dir=/tmp/distbert/tmp_train --overwrite_output_dir \
      --hpu --optimizer=FusedAdamW --hmp --hmp_bf16=./ops_bf16_distilbert_pt.txt --hmp_fp32=./ops_fp32_distilbert_pt.txt --hmp_opt_level=O1 --use_lazy_mode \
      --save_steps=1000 --num_train_epochs=3.0 --world_size=1

   ```
**Run training on 8 HPUs:**

To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo. Number of cards can be configured using --world_size option in the demo script.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).


- 8 HPUs on a single server, BF16, Lazy mode:

   ```bash

   export MASTER_ADDR="localhost"
   export MASTER_PORT="12345"
   mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON run_squad_w_distillation.py \
      --model_type=distilbert --model_name_or_path=distilbert-base-uncased --teacher_type=bert --teacher_name_or_path=bert-base-uncased \
      --config_name=./training_configs/distilbert-base-uncased.json --train_file=/data/pytorch/transformers/Squad/train-v1.1.json \
      --predict_file=/data/pytorch/transformers/Squad/dev-v1.1.json --do_train --do_eval --do_lower_case --output_dir=/tmp/distbert/tmp_train \
      --overwrite_output_dir --hpu --optimizer=FusedAdamW --hmp --hmp_bf16=./ops_bf16_distilbert_pt.txt --hmp_fp32=./ops_fp32_distilbert_pt.txt \
      --hmp_opt_level=O1 --use_lazy_mode --save_steps=500 --num_train_epochs=3.0 --world_size=8

   ```
- 8 HPUs on a single server, FP32, Lazy mode:
   ```bash

   export MASTER_ADDR="localhost"
   export MASTER_PORT="12345"
   mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON run_squad_w_distillation.py \
      --model_type=distilbert --model_name_or_path=distilbert-base-uncased --teacher_type=bert --teacher_name_or_path=bert-base-uncased \
      --config_name=./training_configs/distilbert-base-uncased.json --train_file=/data/pytorch/transformers/Squad/train-v1.1.json \
      --predict_file=/data/pytorch/transformers/Squad/dev-v1.1.json --do_train --do_eval --do_lower_case --output_dir=/tmp/distbert/tmp_train \
      --overwrite_output_dir --hpu --optimizer=FusedAdamW --hmp_opt_level=O1 --use_lazy_mode --save_steps=500 --num_train_epochs=3.0 --world_size=8

   ```


## Supported Configurations

| Device | SynapseAI Version | PyTorch Version |
|-----|-----|-----|
| Gaudi | 1.7.0 | 1.12.0 |

## Changelog
### 1.4.0
 - Removed the redundant mark_step which are no longer needed for distributed training.
 - Instead of setting grads to zero, set the grads to None to improve the performance.
### 1.2.0
 - Set broadcast_buffers = False in torch.nn.parallel.DistributedDataParallel.
 - Enabled HCCL flow for distributed training.

### Training Script Modifications

This section lists the training script, `run_squad_w_distillation.py`, modifications from the original model script.

1. Added Habana Device support.
2. Modifications for saving checkpoint: Bring tensors to CPU and save.
3. Introduced Habana BF16 Mixed precision.
4. Change for supporting HMP disable for optimizer.step.
5. Added support for Fused AdamW optimizer on Habana device for better performance.
6. Added support for Fused Clip Norm for grad clipping on Habana device for better performance.
7. Added changes to support,  mode wit required mark_step.
8. Added smoothing=1 to tqdm for average performance report.

Below are multi-card specific changes:

9. Add lazy mode mark_step before training w/a for distributed training.
10. Modified training script to use mpirun for distributed training.
11. Gradients are used as views using gradient_as_bucket_view.
12. Default allreduce bucket size set to 230MB for better performance in distributed training.
13. Changes for dynamic loading of HCCL library.
14. All required environmental variables brought under training script for ease of usage.
15. Make sure only one process printout logs from this script


## Citation

@inproceedings{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  booktitle={NeurIPS EMC^2 Workshop},
  year={2019}
}

