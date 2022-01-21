# DistilBert for PyTorch

# Table of Contents

 [Model Overview](#model-overview)

 [Setup](#setup)

 [SQuADv1.1 dataset preparation](#squadv11-dataset-preparation)

 [Fine Tuning](#fine-tuning)

 [Training Examples](#training-examples)

 [Training Script Modifications](#training-script-modifications)

 [Known Issues](#known-issues)

 [Citation](#citation)

### Model Overview

Distil* is a class of compressed models that started with DistilBERT. DistilBERT stands for Distilled-BERT. DistilBERT is a small, fast, cheap and light Transformer model based on Bert architecture. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving 97% of BERT's performances as measured on the GLUE language understanding benchmark. DistilBERT is trained using knowledge distillation, a technique to compress a large model called the teacher into a smaller model called the student. By distillating Bert, we obtain a smaller Transformer model that bears a lot of similarities with the original BERT model while being lighter, smaller and faster to run. DistilBERT is thus an interesting option to put large-scaled trained Transformer model into production.

This repository is a copy of the distillation scripts on the original Huggingface repository (<https://github.com/huggingface/transformers.git>) at transformers/examples/research_projects/distillation/, commit 3b1f5caff26c08dfb74a76de1163f4becde9e828 from June 11, 2021.

This folder contains Squad fine-tune DistilBERT model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy. Please visit [this page](https://developer.habana.ai/resources/habana-training-models/#performance) for performance information.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

### Setup

Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. (Run the hl-smi utility to determine the SynapseAI version.)

```bash
 git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

### SQuADv1.1 dataset preparation

Public datasets available on the datasets hub at <https://github.com/huggingface/datasets>.

## Fine Tuning

The current script only has teacher_type bert case fully tested.

* teacher_type Bert fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in eager mode.
* teacher_type Bert fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in Lazy mode.

The Demo script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.


* Located in: Model-References/PyTorch/nlp/distilbert/
* Suited for task ***squad:*** Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
* Uses optimizer: **AdamW** (“ADAM with Weight Decay Regularization”).
* Light-weight
* Datasets for SQuAD needs to be downloaded to the docker container.

## Training Examples

### Single Card Training Examples

1. Fine-tune DistilBert with teacher_type bert (eager mode)

* Run DistilBert using FP32 data type:

```

$PYTHON demo_distilbert.py finetuning --model_name_or_path distilbert-base-uncased --mode eager --teacher_type "bert" --do_eval --train_file <SQuAD dataset path>/train-v1.1.json --predict_file <SQuAD dataset path>/dev-v1.1.json --data_type fp32

```

* Run DistilBert using BF16 data type:

```

$PYTHON demo_distilbert.py finetuning --model_name_or_path distilbert-base-uncased --mode eager --teacher_type "bert" --do_eval --train_file <SQuAD dataset path>/train-v1.1.json --predict_file <SQuAD dataset path>/dev-v1.1.json --data_type bf16

```

2. Fine-tune DistilBert with teacher_type bert (lazy mode)

* Run DistilBert using FP32 data type:
```

$PYTHON demo_distilbert.py finetuning --model_name_or_path distilbert-base-uncased --mode lazy --teacher_type "bert" --do_eval --train_file <SQuAD dataset path>/train-v1.1.json --predict_file <SQuAD dataset path>/dev-v1.1.json --data_type fp32

```
* Run DistilBert using BF16 data type:

```

$PYTHON demo_distilbert.py finetuning --model_name_or_path distilbert-base-uncased --mode lazy --teacher_type "bert" --do_eval --train_file <SQuAD dataset path>/train-v1.1.json --predict_file <SQuAD dataset path>/dev-v1.1.json --data_type bf16

```

### Multicard Training Examples


To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo. Number of cards can be configured using --world_size option in the demo script.

Use the following command to run the multicard demo on 8 cards (1 server) for bf16 Lazy mode:
```

$PYTHON demo_distilbert.py finetuning --model_name_or_path distilbert-base-uncased --mode lazy --teacher_type "bert" --do_eval --train_file <SQuAD dataset path>/train-v1.1.json --predict_file <SQuAD dataset path>/dev-v1.1.json --data_type bf16 --world_size 8

```
Use the following command to run the multicard demo on 8 cards (1 server) for fp32 Lazy mode:
```

$PYTHON demo_distilbert.py finetuning --model_name_or_path distilbert-base-uncased --mode lazy --teacher_type "bert" --do_eval --train_file <SQuAD dataset path>/train-v1.1.json --predict_file <SQuAD dataset path>/dev-v1.1.json --data_type fp32 --world_size 8

```

### Training Script Modifications

This section lists the training script run_squad_w_distillation.py modifications from the original model script.

1. Added Habana Device support.
2. Modifications for saving checkpoint: Bring tensors to CPU and save.
3. Introduced Habana BF16 Mixed precision.
4. Change for supporting HMP disable for optimizer.step.
5. Added support for Fused AdamW optimizer on Habana device for better performance.
6. Added support for Fused Clip Norm for grad clipping on Habana device for better performance.
7. Added changes to support Lazy mode with required mark_step.
8. Added smoothing=1 to tqdm for average performance report.

Below are multicard specific changes

9. Add lazy mode mark_step before training w/a for distributed training.
10. Modified training script to use mpirun for distributed training.
11. Gradients are used as views using gradient_as_bucket_view.
12. Default allreduce bucket size set to 230MB for better performance in distributed training.
13. Changes for dynamic loading of HCCL library.
14. All required enviornmental variables brought under training script for ease of usage.
15. Make sure only one process printout logs from this script

# Changelog
## 1.2.0
 - Set broadcast_buffers = False in torch.nn.parallel.DistributedDataParallel
 - Enabled HCCL flow for distributed training
### Known Issues


### Citation

@inproceedings{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  booktitle={NeurIPS EMC^2 Workshop},
  year={2019}
}

