# Running Intel-HabanaLabs MLPerf™ GPT3 Benchmark on 1024 Gaudi2 cards

This directory provides instructions to reproduce Intel-HabanaLabs's results for MLPerf Training v4.0 submission of GPT3 model on 128 servers with 8 Gaudi2 cards each.

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/)

MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use is strictly prohibited.

- [Running Intel-HabanaLabs MLPerf™ GPT3 Benchmark on 1024 Gaudi2 cards](#running-intel-habanalabs-mlperf-gpt3-benchmark-on-1024-gaudi2-cards)
  - [Setup](#setup)
    - [Prepare MLPerf Directory](#prepare-mlperf-directory)
    - [Build and Deploy Intel-HabanaLabs 1.14.0 Container](#build-and-deploy-intel-habanalabs-1140-container)
    - [Training Data for GPT3-175B](#training-data-for-gpt3-175b)
      - [Dataset Preparation for GPT3-175B](#dataset-preparation-for-gpt3-175b)
      - [Checkpoint Preparation for GPT3-175B](#checkpoint-preparation-for-gpt3-175b)
  - [Training GPT3-175B](#training-gpt3-175b)
    - [Installing Requirements](#installing-requirements)
    - [Run and time](#run-and-time)
      - [Running GPT3 on 1024 Gaudi2 cards](#running-gpt3-on-1024-gaudi2-cards)

## Setup

### Prepare MLPerf Directory

On each compute node, perform the following:

1. Follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/v1.14.0/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the benchmarks on Gaudi.

2. Create directories for logs and dataset folders:
    ```
    export MLPERF_DIR=/path/to/mlperf/root
    export LOG_DIR=$MLPERF_DIR/logs
    export DATASETS_DIR=$MLPERF_DIR/datasets
    mkdir -p $MLPERF_DIR/Intel-HabanaLabs $LOG_DIR $DATASETS_DIR
    ```

    **Note:** It is essential to place the `$MLPERF_DIR` on a shared filesystem that is accessible by all the nodes. This allows for dataset preparation to be performed only once in the `Training Data for <configuration>` sections, enabling all nodes to access the prepared dataset during training, as well as model's code.

3. This README is located in `benchmarks/gpt3` directory corresponding to Intel-HabanaLabs's gpt3 submission.
Download this whole `benchmarks` folder along with all subfolders and copy it under `$MLPERF_DIR/Intel-HabanaLabs`

### Build and Deploy Intel-HabanaLabs 1.14.0 Container

To build MLPerf the container, perform the following:

1. Set docker image:
    ```
    export DOCKER_IMAGE=vault.habana.ai/gaudi-docker/1.14.0/ubuntu22.04/habanalabs/pytorch-installer-2.1.1:1.14.0-493-20240424
    ```

2. Create `mlperf4.0` container by running the following command.

    ```bash
    export CONTAINER_NAME=mlperf4.0
    docker run --privileged --security-opt seccomp=unconfined \
      --name $CONTAINER_NAME -td                              \
      -v /dev:/dev                                            \
      --device=/dev:/dev                                      \
      -e LOG_LEVEL_ALL=6                                      \
      -v /sys/kernel/debug:/sys/kernel/debug                  \
      -v /tmp:/tmp                                            \
      -v $MLPERF_DIR:/root/MLPERF                             \
      -v $DATASETS_DIR:/root/datasets                         \
      -v $LOG_DIR:/root/logs                                  \
      --cap-add=sys_nice --cap-add=SYS_PTRACE                 \
      --user root --workdir=/root --net=host                  \
      --ulimit memlock=-1:-1 ${DOCKER_IMAGE}
    ```

3. Start the docker.

    ```bash
    docker exec $CONTAINER_NAME bash -c "service ssh start"
    docker exec -it $CONTAINER_NAME bash
    ```

4. In the docker, create `/root/shared/hosts` file that contains a list of all host IPs in the cluster. Add one IP per line. Below is an example for 4 nodes (32 devices).
    ```
    mkdir /root/shared
    echo '10.10.100.101' > /root/shared/hosts
    echo '10.10.100.102' >> /root/shared/hosts
    echo '10.10.100.103' >> /root/shared/hosts
    echo '10.10.100.104' >> /root/shared/hosts
    ```

5. SSH is used to spawn local and remote processes. In order to allow communication between machines it is required to provide a passwordless _ssh_ communication and set default port for connection. It has to be done on all of the machines:
    ```
    mkdir .ssh
    printf 'Host *\n    StrictHostKeyChecking no\nPort 3022' >> .ssh/config
    ```
    It also may be necessary to setup SSH keys and add them to `~/.ssh/authorized_keys`.


### Training Data for GPT3-175B

#### Dataset Preparation for GPT3-175B

Dataset preparation should be done in the following docker:

```
docker run --ipc=host -it -v $DATASETS_DIR:/root/datasets -v $MLPERF_DIR:/root/MLPERF nvcr.io/nvidia/pytorch:22.11-py3 bash
```

MLPerf GPT3 is trained using C4/en/3.0.1 dataset. It can be downloaded from https://huggingface.co/datasets/allenai/c4. Instruction is clear on how to select precisely the files for downloading.

```
apt-get update
apt-get install git-lfs
mkdir -p /root/datasets/gpt3
cd /root/datasets/gpt3
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"
```

Out of all the files, only 256 will be required for training, and 8 for validation.
You can merge them into three .json.gz files using the following commands, which are taken from https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/README.md.

```
# create softlinks to store each shard before merging
mkdir -p softlinks
for shard in {6..7}; do
  start=$((shard * 128))
  end=$((shard * 128 + 127))
  mkdir -p softlinks/en_$shard
  for ind in $(seq -f "%05g" $start $end); do
    ln -s ../../en/c4-train.${ind}-of-01024.json.gz softlinks/en_${shard}/c4-train.${ind}-of-01024.json.gz
  done
done

# merge
mkdir -p en_merge
for shard in {6..7}; do
  cat softlinks/en_${shard}/*gz > en_merge/c4-train.en_${shard}.json.gz
done
cat en/c4-validation.0000* > en_merge/c4-validation.json.gz
```

To tokenize the prepared files, you need to download the tokenizer model, vocab_c4_en_301_5Mexp2_spm.model, and the vocabulary file, vocab_c4_en_301_5Mexp2_spm.vocab, from the following location:
https://console.cloud.google.com/storage/browser/mlperf-llm-public2;tab=objects?prefix=&forceOnObjectsSortingFiltering=false.
Please note that registration is required to access these files. Tokenization can be performed using the following commands.
Please be aware that this conversion process may take several hours.

```
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && git checkout f3ad584b94170bc3ea197df29eb9ef9c96061730 && bash ./reinstall.sh && cd ..

mkdir -p preprocessed_c4_spm
for shard in {6..7}; do
python3 NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input en_merge/c4-train.en_${shard}.json.gz \
    --tokenizer-library sentencepiece \
    --tokenizer-model vocab_c4_en_301_5Mexp2_spm.model \
    --output-prefix preprocessed_c4_spm/c4_en_${shard}_c4_spm \
    --dataset-impl mmap \
    --workers 128
done

python3 NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input en_merge/c4-validation.json.gz \
    --tokenizer-library sentencepiece \
    --tokenizer-model vocab_c4_en_301_5Mexp2_spm.model \
    --output-prefix preprocessed_c4_spm/c4_en_validation_c4_spm \
    --dataset-impl mmap \
    --workers 128
```

The resulting files to be used during training are as follows:
* ```preprocessed_c4_spm/c4_en_6_c4_spm_text_document.bin```
* ```preprocessed_c4_spm/c4_en_6_c4_spm_text_document.idx```
* ```preprocessed_c4_spm/c4_en_7_c4_spm_text_document.bin```
* ```preprocessed_c4_spm/c4_en_7_c4_spm_text_document.idx```
* ```preprocessed_c4_spm/c4_en_validation_c4_spm_text_document.bin```
* ```preprocessed_c4_spm/c4_en_validation_c4_spm_text_document.idx```

In addition to the dataset, GPT3 implementation requires https://huggingface.co/gpt2/resolve/main/vocab.json and https://huggingface.co/gpt2/resolve/main/merges.txt files:

```
wget "https://huggingface.co/gpt2/resolve/main/vocab.json" -P preprocessed_c4_spm
wget "https://huggingface.co/gpt2/resolve/main/merges.txt" -P preprocessed_c4_spm
```

In order to exclude graph compilation time from Time To Train, you need to prepare a synthetic dataset for device warmup:
```
python3 /root/MLPERF/Intel-HabanaLabs/benchmarks/gpt3/tools/create_synthetic_dataset.py \
    --valid_files_path preprocessed_c4_spm/c4_en_validation_c4_spm_text_document \
    --output_path preprocessed_c4_spm/
```

The commandline above will create synthetic files:
* ```preprocessed_c4_spm/synthetic_text_document.bin```
* ```preprocessed_c4_spm/synthetic_text_document.idx```

#### Checkpoint Preparation for GPT3-175B

At one stage, there will be a megatron checkpoint directory and a universal checkpoint directory, each requiring 2 TB of disk space. Therefore, to complete all the steps, it is necessary to have over 4TB of free disk space.
Additionally, the machine must have a minimum of 32 CPUs and 755GB of RAM to convert the checkpoint.

Log into mlperf4.0 PyTorch container. Install DeepSpeed and other requirements:
```
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.14.0
pip install -r /root/MLPERF/Intel-HabanaLabs/benchmarks/gpt3/requirements.txt
```

1. Download paxml checkpoint from S3 bucket:

   Install rclone:
   ```
   sudo -v ; curl https://rclone.org/install.sh | sudo bash
   ```

   Configure rclone for s3 bucket:
   ```
   rclone config create mlc-training s3 provider=Cloudflare access_key_id=76ea42eadb867e854061a1806220ee1e secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
   ```

   Download the paxml checkpoint to desired location e.g. `/root/datasets`:
   ```
   rclone copy mlc-training:mlcommons-training-wg-public/gpt3/paxml /root/datasets -P
   ```

2. Convert the paxml checkpoint to Megatron checkpoint:

    ```
    python3 /root/MLPERF/Intel-HabanaLabs/benchmarks/gpt3/tools/convert_checkpoint/convert_paxml_optimizer.py \
            --google_ckpts /root/datasets/paxml/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoint_00004000/ \
            --output_dir /root/datasets/megatron_merged_ckpt \
            --num_layers 96 \
            --params_file /root/MLPERF/Intel-HabanaLabs/benchmarks/gpt3/tools/convert_checkpoint/common_bf16.json \
            --pool 1
    ```

3. Convert Megatron merged checkpoint to DeepSpeed universal.

    To generate the mp-rank-files required in megatron_optim_merged_to_ds_universal_convert.py, the user needs to run GPT-3, which will generate these files based on the configuration used in the run.
    This can be obtained by running a single step of GPT-3 and saving the checkpoint.
    Please note that only this particular step of checkpoint peparation must be done using 8 HLS2 machines. The remaining steps can be performed on a CPU-only machine.
    Please make sure /root/shared/hosts file contains a list of 8 IPs for HLS2 machines and SSH communication is properly configured.
    For further details, refer to points 4 and 5 [here](#build-and-deploy-intel-habanalabs-1140-container).
    Once the setup is ready, proceed to run the single step for GPT3 as follows:
    ```
    mkdir checkpoint_with_mp_rank_files
    bash /root/MLPERF/Intel-HabanaLabs/benchmarks/gpt3/run_gpt.sh --hosts /root/shared/hosts --data-dir /root/datasets/ --output-dir /root/logs --num-nodes 8 --data-parallel-size 1 --start-from-ckpt false --save-checkpoints-dir checkpoint_with_mp_rank_files --exit-interval 1 --global-batch-size 2048
    ```

    Run megatron_optim_merged_to_ds_universal_convert.py to create the universal checkpoint:

    ```
    mkdir -p /root/datasets/gpt3/universal-checkpoint
    python3 /root/MLPERF/Intel-HabanaLabs/benchmarks/gpt3/tools/convert_checkpoint/megatron_optim_merged_to_ds_universal_convert.py \
        --o /root/datasets/gpt3/universal-checkpoint/ --ds-mp-rank-files-dir checkpoint_with_mp_rank_files --megatron-lm-merged-input-dir /root/datasets/megatron_merged_ckpt \
        --tp 8 --pp 8 --nl 96 --iteration 3000 --global-batch-size 2048 --seq_length 2048 --lr-decay-samples 166809600 --lr-warmup-samples 407040 \
        --pool 64 --model-parallel-same-config False --update-only-mp-rank-files False
    ```

## Training GPT3-175B

All the training steps for GPT3-175B should be performed in mlperf4.0 PyTorch container.

### Installing Requirements

The following requirements need to be installed on all machines participating in the training:
```
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.14.0
pip install -r /root/MLPERF/Intel-HabanaLabs/benchmarks/gpt3/requirements.txt
```

### Run and time

The latest Intel-HabanaLabs's software supports 8-bit floating-point precision (FP8) training for GPT3 model and MLPerf4.0 submissions for GPT3 have been conducted using FP8 precision.
Running the GPT3 model requires multiple machines. For example, 128 HLS2 machines (1024 Gaudi2 cards).

Please set the paths for the dataset and the universal checkpoint, which should be created during [setup phase](#training-data-for-gpt3-175b).
```
export DATASET_DIR=/root/datasets/gpt3/c4/preprocessed_c4_spm
export CHECKPOINT_DIR=/root/datasets/gpt3/universal-checkpoint
```

Please make sure /root/shared/hosts file contains a list of IPs for HLS2 machines, and that SSH communication is properly configured.
For further details, refer to points 4 and 5 [here](#build-and-deploy-habanalabs-mlperf-training-31-container).

#### Running GPT3 on 1024 Gaudi2 cards
```
bash /root/MLPERF/Intel-HabanaLabs/benchmarks/gpt3/run_gpt.sh --data-dir $DATASET_DIR/ --universal-ckpt-path $CHECKPOINT_DIR/ \
--hosts /root/shared/hosts --output-dir /root/logs --num-nodes 128 --data-parallel-size 16 --pipeline-model-parallel-size 8 --tensor-model-parallel-size 8 --save-checkpoints false --mllog-output-path /root/logs/result.txt --train-samples 6782976 --global-batch-size 2048 --micro-batch-size 2 --eval-interval 12 --device-warmup true --device-warmup-dataset-path $DATASET_DIR/synthetic_text_document --use-fp8-transformer-engine --use-fused-sdpa-with-recompute true --log-interval 12
```

Training results will be stored in `/root/logs` folder.

The script will start from universal checkpoint and train up to 312 steps or the time, when validation log perplexity is below 2.69. According to the convergence point of GPT3 on HLS system, it should approximately run for 288 steps in order to reach 2.69 validation log perplexity. To reduce number of steps, you can use `--exit-interval` parameter or reduce train samples by `--train-samples` parameter.
