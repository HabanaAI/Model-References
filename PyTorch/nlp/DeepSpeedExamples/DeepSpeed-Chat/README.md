# DeepSpeed-Chat: RLHF Training of Chat Models

This directory provides scripts for the 3 stages RLHF training of LM to a chat model.
The implementation is based on [Microsoft/DeepSpeedExamples Commit 8f8099a](https://github.com/microsoft/DeepSpeedExamples/tree/8f8099a813f3b223d5df39e0c15c748de4eb1669/applications/DeepSpeed-Chat)

## Table of Contents
* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)
* [Known Issues](#knownissues)

## Model Overview
The training process involves 3 stages as described in [Microsoft/DeepSpeedExample README file](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat#-demonstration-individual-step-fine-tuning).
The below model are used:
1. Step 1 - Supervised Fine-Tuning: in this stage we used the pre-trained bigscience/bloom-1b1.
2. Step 2 - Reward Model: in this stage we used pre-trained bigscience/bloom-560m.
3. Step 3 - Reinforcement Learning with Human Feedback: Used the fined tuned models from Step-1 and Step-2.

## Setup
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi2.

### Install Habana DeepSpeed-fork
Please follow the instruction in [DeepSpeed User Guide](https://docs.habana.ai/en/master/PyTorch/DeepSpeed/DeepSpeed_User_Guide/DeepSpeed_User_Guide.html)

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that matches your SynapseAI version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.
```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

```
export MODEL_REFERENCES_ROOT=/path/to/Model-References
export PYTHONPATH=/path/to/Model-References/PyTorch/common:$PYTHONPATH

```

### Install Model Requirements
* In the docker container, go to the model directory:
  ```bash
  cd Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/
  ```

* Install the required packages using pip:
  ```bash
  pip install -r requirements.txt
  ```

## Training and Examples
Example bash script for Steps 1, 2 and 3 on single and multi-card setups are available under `Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/scripts/bloom/refs`
The below READMEs further explain each bash script:
* [Step 1](training/step1_supervised_finetuning/README.md)
* [Step 2](training/step2_reward_model_finetuning/README.md)
* [Step 3](training/step3_rlhf_finetuning/README.md)

## Supported Configuration
| Validated on  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|-------------|
| Gaudi2  | 1.12.0           | 2.0.1          | Training |


## Changelog
### 1.11.0
* Introduce this training script.

### Script Modifications
- Cloned DeepSpeed-Chat directory from [Microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) at commit 8f8099a
- Added support for HPU accelerator.
- Fixed tokenizer for non OPT models.
- Fixed chat application.
- Added backward support for DeepSpeed v0.8.3
- Added support for tensorboard logging during training.
- Added support for BF16 training.
- Added support for explicit dropout configuration.
- Added priodic evaluation during Reward Model training.
- Converted the Reward Model loss from BF16 to Float.
- Fixed weight decay configuration for Bloom models.
- Fixed step-2 accuracy for bloom-560m by restting the rwtranformer.ln_f weights.
- Added reward scopre EMA in step3.
- Added option for read only cached dataset storage.
- Added an option to load tokenizer from local storage.
- Optimized RewardModel loss calculation to avoid dynamic shapes and non-infereable ops.
- Added an option to use non fused optimizer.
- Fixed step-1 PPL calculation.
- Changed step-1, 2 and 3 loss calculation to FP32.
- Improved step-3 perofrmance by using optimum-habana generate function instead of the transformers implementation in case of HPU accelerator.
- Added an option to use HPUGraph as performance optimization during generate to avoid host overhead for inference.

## Known Issues
- Step3 of Chat model finetuning using RLHF may achieve lower accuracy than expected.