# Reinforcement Learning from human feedback (RLHF) finetuning
For more back ground you can visit https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning .
This page will provide basic instructions on how the run step-3 RLHF script using single and multicard setups.

## Example Script
The example bash script to launch step-3 training is located in `Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/example_scripts/train_step3_bloom_1.1b_560m.sh`

The script respects the below environment variables to control the training:
- `HL_TAG`: tag name added to the artifacts of this run (string) - Mandatory.
- `HL_BASE_OUT_PATH`: base path for artifacts - Mandatory.
- `HL_ACTOR_MODEL_PATH`: path for actor model checkpoint from step1 - Mandatory.
- `HL_CRITIC_MODEL_PATH` - path for critic model checkpoint from step2 - Mandatory.
- `HL_DATASET_PATH` - HF like dataset path or list of paths, mandatory
- `HL_NUM_NODES`: Number of "boxes" (servers) participating in the training process (currently supporting 1 node) - Mandatory.
- `HL_DEVICES_PER_NODE`: number of HPU accelarator cards per node - Mandatory.
- `HL_ACTOR_CP_ACT`: whether to use activation-checkpointing memory optimization for actor model - Optional, Default is set to 0 (false).
- `HL_CRITIC_CP_ACT`: whether to use activation-checkpointing memory optimization for critic model - Optional, Default is set to 0 (false).
- `HL_SEED`: base seed that will be used to system initialization - Optional, default set to 10.
- `HL_MBS`: the micro-bs that each card will use during the training - Optional, default is 2.
- `HL_TENSORBOARD_PATH`: tensorboard path - Optional, empty string for default.
- `HL_LOG_FILE`: log full filename- Optional, empty string for default
- `HL_MASTER_PORT`: deepspeed runner master_port - Optional, default is 29500

It can be called with the below template:
  ```bash
  cd Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/
  export HL_ACTOR_MODEL_PATH=<act_model_path>
  export HL_CRITIC_MODEL_PATH=<cri_model_path>
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  ...
  ./scripts/bloom/refs/train_step3_bloom_1.1b_560m
  ```

## Single card execution example
The below script execution will use 1 node, and 1 HPU card:
  ```bash
  cd Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/
  export HL_TAG=step3_single_card
  export HL_ACTOR_MODEL_PATH=<act_model_path>
  export HL_CRITIC_MODEL_PATH=<cri_model_path>
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  export HL_NUM_NODES=1
  export HL_DEVICES_PER_NODE=1
  ...
  ...
  ./scripts/bloom/refs/train_step3_bloom_1.1b_560m
  ```

## Multicard card execution example
The below script execution will use 1 node, and 8 HPU cards:
  ```bash
  cd Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/
  export HL_TAG=step3_x8_cards
  export HL_ACTOR_MODEL_PATH=<act_model_path>
  export HL_CRITIC_MODEL_PATH=<cri_model_path>
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  export HL_NUM_NODES=1
  export HL_DEVICES_PER_NODE=8
  ...
  ...
  ./scripts/bloom/refs/train_step3_bloom_1.1b_560m
  ```
  