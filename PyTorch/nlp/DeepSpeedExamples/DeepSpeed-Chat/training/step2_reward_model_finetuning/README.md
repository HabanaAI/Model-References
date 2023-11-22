# Reward Model (RM) finetuning
For more back ground you can visit https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning .
This page will provide basic instructions on how the run step-2 Finetuning script using single and multicard setups.

## Example Script
The example bash script to launch step-2 training is located in `Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/example_scripts/train_step2_bloom_560m.sh`

The script respects the below environment variables to control the training:
- `HL_TAG`: tag name added to the artifacts of this run (string) - Mandatory.
- `HL_BASE_OUT_PATH`: base path for artifacts - Mandatory.
- `HL_NUM_NODES`: Number of "boxes" (servers) participating in the training process (currently supporting 1 node) - Mandatory.
- `HL_DEVICES_PER_NODE`: number of HPU accelarator cards per node - Mandatory.
- `HL_CRITIC_ZERO_STAGE`: The zero stage DeepSpeed will use (setting to 0, will direct to BF16Optimizer which implements Zero1 natively). 
- `HL_SEED`: base seed that will be used to system initialization - Optional, default set to 10.
- `HL_MBS`: the micro-bs that each card will use during the training - Optional, default is 8.
- `HL_TENSORBOARD_PATH`: tensorboard path - Optional, empty string for default.
- `HL_LOG_FILE`: log full filename- Optional, empty string for default
- `HL_MASTER_PORT`: deepspeed runner master_port - Optional, default is 29500
- `HL_CRITIC_MODEL`: The path from which to load the pretrained model, can be also HF based path. by default will try to load bigscience/bloom-560m
- `HL_DATASET_PATH` - HF like dataset path or list of paths, mandatory

It can be called with the below template:
  ```bash
  cd Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/
  export HL_TAG=<tag>
  export HL_BASE_OUT_PATH=<base_out_path>
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  ...
  ...
  ./scripts/bloom/refs/train_step2_bloom_560m.sh
  ```

## Single card execution example
The below script execution will use 1 node, and 1 HPU card:
  ```bash
  cd Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/
  export HL_TAG=stage2_single_card
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  ...
  ...
  export HL_NUM_NODES=1
  export HL_DEVICES_PER_NODE=1
  ...
  ./scripts/bloom/refs/train_step2_bloom_560m.sh
  ```

## Multicard card execution example
The below script execution will use 1 node, and 8 HPU cards:
  ```bash
  cd Model-References/PyTorch/nlp/DeepSpeedExamples/DeepSpeed-Chat/
  export HL_TAG=stage2_x8_cards
  export HL_DATASET_PATH=<path_to_data_set_or_list>
  ...
  ...
  export HL_NUM_NODES=1
  export HL_DEVICES_PER_NODE=8
  ...
  ./scripts/bloom/refs/train_step2_bloom_560m.sh
  ```
  