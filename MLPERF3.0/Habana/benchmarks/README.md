# Running Habana MLPerf™ Benchmarks

This directory provides instructions to reproduce Habana's results for [MLPerf Training v3.0](https://habana.ai/since-habanas-last-mlperf-submission/) **on 1 to 48 servers configurations with 8 Gaudi2 cards each.**

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/)

MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use is strictly prohibited.

- [Running Habana MLPerf™ Benchmarks](#running-habana-mlperf-benchmarks)
  - [Setup](#setup)
    - [Prepare MLPerf Directory](#prepare-mlperf-directory)
    - [Build and Deploy HabanaLabs MLPerf Training 3.0 Container](#build-and-deploy-habanalabs-mlperf-training-30-container)
    - [Training Data for TensorFlow BERT](#training-data-for-tensorflow-bert)
    - [Training Data for PyTorch BERT](#training-data-for-pytorch-bert)
    - [Training Data for ResNet50](#training-data-for-resnet50)
    - [Training Data for Unet3D](#training-data-for-unet3d)
    - [Training Data for GPT3-175B](#training-data-for-gpt3-175b)
  - [Training BERT](#training-bert)
  - [Training ResNet50](#training-resnet50)
  - [Training Unet3D](#training-unet3d)
  - [Training GPT3-175B](#training-gpt3-175b)
  - [Supported Configurations](#supported-configurations)
  - [Changelog](#changelog)

## Setup

### Prepare MLPerf Directory

On each compute node, perform the following:

1. Follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the benchmarks on Gaudi.

2. Create directories for scratch and dataset folders:
    ```
    export MLPERF_ROOT=/path/to/mlperf/root
    export SCRATCH_DIR=$MLPERF_ROOT/scratch
    export DATASETS_DIR=$MLPERF_ROOT/datasets
    mkdir -p $SCRATCH_DIR
    mkdir -p $DATASETS_DIR
    ```

    **Note:** If training is to be conducted on multiple nodes, it is essential to place the $DATASETS_DIR on a shared filesystem that is accessible by all the nodes. This allows for dataset preparation to be performed only once in the `Training Data for <configuration>` sections, enabling all nodes to access the prepared dataset during training.

3. Clone Model-References repository and switch to the branch that matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

    ```bash
    cd $MLPERF_ROOT
    git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
    export MLPERF_DIR=$MLPERF_ROOT/Model-References/MLPERF3.0
    ```

### Build and Deploy HabanaLabs MLPerf Training 3.0 Container

To build MLPerf training 3.0 container, perform the following:

1. Copy ssh keys to enable passwordless ssh to /root/.ssh/
2. Set the environment variables for the docker command.
   * To find a docker image, go to [gaudi-docker](https://vault.habana.ai/ui/repos/tree/General/gaudi-docker).
   * Open gaudi-docker directory, and select the folder that matches the SynapseAI version (determined by running [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)).
   * Navigate to subdirectories, choose system and framework version.
   * Choose the docker build version. Most often 'latest' will be used.
   * Navigate to "Docker Info" tab and note "Title" string.
   * Set `DOCKER_IMAGE` to "Title" string with `vault.habana.ai/gaudi-docker/` prefix. See the examples below.
     * Example on TensorFlow Container:
          ```bash
          # NOTE: The below is only an example value. Replace [SynapseAI version] and [TF version] to match your setup and Supported Configuration.
          export DOCKER_IMAGE=vault.habana.ai/gaudi-docker/[SynapseAI version]/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-[TF version]:latest
          export CONTAINER_NAME=mlperf3_0
          ```
      * Example on PyTorch Container:
          ```bash
          # NOTE: The below is only an example value. Replace [SynapseAI version] and [PT version] to match your setup and Supported Configuration.
          export DOCKER_IMAGE=vault.habana.ai/gaudi-docker/[SynapseAI version]/ubuntu20.04/habanalabs/pytorch-installer-[PT Version]:latest
          export CONTAINER_NAME=mlperf3_0
          ```


3. Create `mlperf3_0 container` by running the following command.

    ```bash
    docker run --privileged --security-opt seccomp=unconfined \
      --name $CONTAINER_NAME -td                              \
      -v /dev:/dev                                            \
      --device=/dev:/dev                                      \
      -e LOG_LEVEL_ALL=6                                      \
      -v /sys/kernel/debug:/sys/kernel/debug                  \
      -v /tmp:/tmp                                            \
      -v $MLPERF_DIR:/root/MLPERF                             \
      -v $DATASETS_DIR:/root/datasets                         \
      -v $SCRATCH_DIR:/root/scratch                           \
      --cap-add=sys_nice --cap-add=SYS_PTRACE                 \
      --user root --workdir=/root --net=host                  \
      --ulimit memlock=-1:-1 ${DOCKER_IMAGE}
    ```

4. Start the docker.

    ```bash
    docker exec $CONTAINER_NAME bash -c "service ssh start"
    docker exec -it $CONTAINER_NAME bash
    ```

    **Note:** The following two steps are only necessary for training on multiple nodes.

5. In the docker, create `/root/shared/hosts` file that contains a list of all host IPs in the cluster. Add one IP per line. Below is an example for 4 nodes (32 devices).
    ```
    mkdir /root/shared
    echo '10.10.100.101' > /root/shared/hosts
    echo '10.10.100.102' >> /root/shared/hosts
    echo '10.10.100.103' >> /root/shared/hosts
    echo '10.10.100.104' >> /root/shared/hosts
    ```

6. SSH is used to spawn local and remote processes. In order to allow communication between machines it is required to provide a passwordless _ssh_ communication and set default port for connection. It has to be done on all of the machines:
    ```
    mkdir .ssh
    printf 'Host *\n    StrictHostKeyChecking no\nPort 3022' >> .ssh/config
    ```
    It also may be necessary to setup SSH keys and add them to `~/.ssh/authorized_keys`.

### Training Data for TensorFlow BERT

1. Log into MLPERF3.0 TensorFlow container and install the requirements:
    <!-- DATASET download_mlperf_bert_tensorflow -->
    <!-- DATASET process_mlperf_bert_tensorflow -->
    ```bash
    export BERT_PATH=/root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert
    cd $BERT_PATH
    pip install -r requirements.txt
    ```
    <!-- /DATASET process_mlperf_bert_tensorflow -->
    <!-- /DATASET download_mlperf_bert_tensorflow -->

2. Download the required files from Google drives.
    <!-- DATASET download_mlperf_bert_tensorflow -->
    ```bash
    export TENSORFLOW_BERT_DATA=/root/datasets/tensorflow_bert
    bash pretraining/prepare_dataset.sh \
      --data-path $TENSORFLOW_BERT_DATA \
      --only-download
    ```
    <!-- /DATASET download_mlperf_bert_tensorflow -->

    After completing this step, there should be a `$TENSORFLOW_BERT_DATA/input` folder containing the following files:
    ```
    bert_config.json
    model.ckpt-28252.data-00000-of-00001
    model.ckpt-28252.index
    model.ckpt-28252.meta
    results_text.tar.gz
    vocab.txt
    ```

3. Prepare the packed dataset by running the command below:
    <!-- DATASET process_mlperf_bert_tensorflow -->
    ```bash
    bash pretraining/prepare_dataset.sh \
      --scripts-path $BERT_PATH \
      --data-path $TENSORFLOW_BERT_DATA \
      --only-preprocessing \
      --jobs-limit 25
    ```
    <!-- /DATASET process_mlperf_bert_tensorflow -->

    This step will take multiple hours to complete.
    The exact time depends on the machine setup and the speed of storage that contains the dataset.
    The `--jobs-limit` option limits the number of pararell processes for converting and packing tfrecords.
    This step is resource consuming,
    and the machine running it must have a minimum of 32 CPUs and 755GB of RAM to ensure proper functioning.

4. `$TENSORFLOW_BERT_DATA` should now contain following folders:
    ```
    checkpoint
    eval_dataset
    input
    packed_data_500
    unpacked_data
    ```

    `input` folder can be removed if the preprocessing has been successfully completed.
    By default, TensorFlow BERT uses only packed data for training,
    as described in the scenario mentioned described [here](#training-for-tensorflow-bert).
    In such cases, the `unpacked_data` is unnecessary and can be deleted.

### Training Data for PyTorch BERT

#### Dataset Preparation

Log into mlperf3.0 PyTorch container and run:
<!-- DATASET download_mlperf_bert_pytorch -->
<!-- DATASET process_mlperf_bert_pytorch -->
```bash
cd /root/MLPERF/Habana/benchmarks/bert/implementations/PyTorch
pip install -r requirements.txt
export PYTORCH_BERT_DATA=/root/datasets/pytorch_bert
```
<!-- /DATASET process_mlperf_bert_pytorch -->
```bash
bash input_preprocessing/prepare_data.sh -o $PYTORCH_BERT_DATA
```
<!-- /DATASET download_mlperf_bert_pytorch -->

At this stage, ```$PYTORCH_BERT_DATA/phase1``` checkpoint and  ```$PYTORCH_BERT_DATA/hdf5/eval_varlength``` evaluation data are ready, while ```$PYTORCH_BERT_DATA/hdf5/training_4320/hdf5_4320_shards_uncompressed``` training data requires packing as described in the following section.

#### Training Data Packing

Once the training data is ready, pack it using a similar code as described in [GraphCore for v1.0 Submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data).

<!-- DATASET process_mlperf_bert_pytorch -->
```bash
mkdir $PYTORCH_BERT_DATA/packed
python3 pack_pretraining_data_pytorch.py \
    --input_dir=$PYTORCH_BERT_DATA/hdf5/training-4320/hdf5_4320_shards_uncompressed \
    --output_dir=$PYTORCH_BERT_DATA/packed \
    --max_predictions_per_seq=76
```
<!-- /DATASET process_mlperf_bert_pytorch -->

For further details, refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027).

### Training Data for ResNet50

The instructions for the ImageNet dataset is applicable for both PyTorch and TensorFlow ResNet50.

 1. Sign up with [image-net.org](http://image-net.org/download-images) and acquire the rights to download original images.
 2. Follow the link to the 2012 ILSVRC and download ILSVRC2012_img_val.tar and ILSVRC2012_img_train.tar.
 Place the files in the folder that will be mapped in mlperf3.0 container (for example, `$DATASETS_DIR`).
 3. Run the script below in mlperf3.0 container (PyTorch or TensorFlow) to unpack the dataset:

    ```
    bash /root/MLPERF/Habana/benchmarks/resnet/scripts/unpack_imagenet.sh \
        --train-archive /path/to/ILSVRC2012_img_train.tar \
        --validation-archive /path/to/ILSVRC2012_img_val.tar \
        --output-path /root/datasets/imagenet \
        --jobs-number 16
    ```

    The script unpacks training and validation packages in parallel.
    In addition, when upacking subarchives from ILSVRC2012_img_train.tar,
    `--jobs-number` defines number of pararell processes allocated for the task.
    Scripts runtime is dependent in large part on the data access speed of the storage where $DATASETS_DIR is located.

### Training Data for Unet3D

The instructions for dataset preparation are based on the original MLCommons guidelines at: https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch#steps-to-download-and-verify-data

Perform following steps without using MLPERF3.0 container:

1. Download the dataset:
    ```
    export UNET3D_PT_DIR=$MLPERF_DIR/Habana/benchmarks/unet3d/implementations/PyTorch
    cd $UNET3D_PT_DIR
    git clone https://github.com/neheller/kits19
    cd kits19
    pip3 install -r requirements.txt
    python3 -m starter_code.get_imaging
    ```
    This will download the original, non-interpolated data to `$UNET3D_PT_DIR/kits19/data`.
2. Build UNet3D docker container for data preprocessing
    ```
    cd $UNET3D_PT_DIR
    docker build -t unet3d-data-preprocessing .
    ```
3. Preprocess the dataset inside UNet3D docker container.

    Note that you need to mount two directories:
   - directory with original, non-interpolated data e.g. `kits19/data`
   - output directory for preprocessed dataset e.g. `kits/preprocessed_data` and mount it to `/root/data`

    ```
    cd $UNET3D_PT_DIR
    mkdir kits19/preprocessed_data
    docker run --ipc=host --rm -it -v $UNET3D_PT_DIR/kits19/data:/root/raw_data -v $UNET3D_PT_DIR/kits19/preprocessed_data/:/root/preprocessed_data  unet3d-data-preprocessing:latest python3 preprocess_dataset.py --data_dir /root/raw_data --results_dir /root/preprocessed_data
    ```
    The command line will preprocess each volume and save it as a numpy array to `$UNET3D_PT_DIR/kits19/preprocessed_data/`. It will also display some statistics like the volume shape, mean and stddev of the voxel intensity. Also, it will run a checksum on each file comparing it with the source.

### Training Data for GPT3-175B

#### Dataset Preparation for GPT3-175B

Dataset preparation should be done in the following docker:

```
docker run --ipc=host -it -v $DATASETS_DIR:/root/datasets nvcr.io/nvidia/pytorch:22.11-py3 bash
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

#### Checkpoint Preparation for GPT3-175B

Log into mlperf3.0 PyTorch container. Install DeepSpeed and other requirements:
```
pip install git+https://github.com/HabanaAI/DeepSpeed.git
pip install -r /root/MLPERF/Habana/benchmarks/gpt3/requirements.txt
```

The checkpoint for MLPerf GPT3 in the paxml format can be downloaded from
[gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000](gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000).
The common_bf16.json can be downloaded from: https://github.com/ShriyaPalsamudram/training/tree/LLM-NVIDIA-reference-draft/large_language_model/megatron-lm/scripts.
At one stage, there will be a merged directory and a universal directory, each requiring 2 TB of disk space for 96L. Therefore, to complete all the steps, it is necessary to have over 4TB of free disk space.
Additionally, the machine must have a minimum of 32 CPUs and 755GB of RAM to ensure proper functioning.
Before the checkpoint can be used, it must be converted by following the steps below:

1. Convert the paxml checkpoint to Megatron distributed using /root/MLPERF/Habana/benchmarks/gpt3/tools/convert_checkpoint/convert_paxml_optimizer.py

    ```
    python3 /root/MLPERF/Habana/benchmarks/gpt3/tools/convert_checkpoint/convert_paxml_optimizer.py \
            --google_ckpts checkpoint_00004000/ \
            --output_dir megatron_merged_ckpt \
            --num_layers 96 \
            --params_file common_bf16.json \
            --pool 1
    ```

2. Convert Megatron merged checkpoint to DeepSpeed universal.

    To generate the mp-rank-files required in megatron_optim_merged_to_ds_universal_convert.py, the user needs to run GPT-3, which will generate these files based on the configuration used in the run.
    This can be obtained by running a single step of GPT-3 and saving the checkpoint.
    Please note that only this particular step of checkpoint peparation must be done using 8 HLS2 machines. The remaining steps can be performed on a CPU-only machine.
    Please make sure /root/shared/hosts file contains a list of 8 IPs for HLS2 machines and SSH communication is properly configured.
    For further details, refer to points 5 and 6 [here](#build-and-deploy-habanalabs-mlperf-training-30-container).
    Once the setup is ready, proceed to run the single step for GPT3 as follows:
    ```
    mkdir checkpoint_with_mp_rank_files
    bash /root/MLPERF/Habana/benchmarks/gpt3/run_gpt.sh --hosts /root/shared/hosts --data-dir /root/datasets/ --output-dir /root/scratch --num-nodes 8 --data-parallel-size 1 --start-from-ckpt false --save-checkpoints-dir checkpoint_with_mp_rank_files --exit-interval 1
    ```

    Run megatron_optim_merged_to_ds_universal_convert.py to create the universal checkpoint:

    ```
    mkdir -p /root/datasets/gpt3/universal-checkpoint
    python3 /root/MLPERF/Habana/benchmarks/gpt3/tools/convert_checkpoint/megatron_optim_merged_to_ds_universal_convert.py \
        --o /root/datasets/gpt3/universal-checkpoint/ --ds-mp-rank-files-dir checkpoint_with_mp_rank_files --megatron-lm-merged-input-dir megatron_merged_ckpt \
        --tp 8 --pp 8 --nl 96 --iteration 4000 --global-batch-size 1536 --seq_length 2048 --lr-decay-samples 166809600 --lr-warmup-samples 407040 \
        --pool 64 --model-parallel-same-config False --update-only-mp-rank-files False
    ```

## Training BERT

### Training TensorFlow BERT

1. Inside the mlperf3.0 TensorFlow container, install BERT requirements.
    ```bash
    export BERT_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/bert/implementations
    pip install -r $BERT_IMPLEMENTATIONS/TensorFlow/nlp/bert/requirements.txt
    ```

2. Run the training.
    ```bash
    cd $BERT_IMPLEMENTATIONS/HLS-Gaudi2-TF
    ./launch_bert_hvd.sh --config defaults.cfg
    ```

### Training PyTorch BERT

1. Inside the mlperf3.0 PyTorch container, install BERT requirements.
    ```bash
    export BERT_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/bert/implementations
    pip install -r $BERT_IMPLEMENTATIONS/PyTorch/requirements.txt
    ```

2. Run the training.
    ```bash
    export PYTORCH_BERT_DATA=/root/datasets/pytorch_bert
    cd $BERT_IMPLEMENTATIONS/HLS-Gaudi2-PT
    ./launch_bert_pytorch.sh --data-dir $PYTORCH_BERT_DATA
    ```

### Training PyTorch BERT on 8 nodes

BERT can also be trained using 8 HLS2 nodes (64 Gaudi2 cards).
Please make sure /root/shared/hosts file contains a list of 8 IPs for HLS2 machines and SSH communication is properly configured.
For further details, refer to 5 and 6 [here](#build-and-deploy-habanalabs-mlperf-training-30-container).

To train PyTorch BERT on 8 nodes, follow the same steps as for [single node PyTorch BERT](#training-for-pytorch-bert). When running 'launch_bert_pytorch.sh', add '-hosts-file' parameter:

```bash
./launch_bert_pytorch.sh --data-dir $PYTORCH_BERT_DATA --hosts-file /root/shared/hosts
```

### TTT (Time to Train) Calculation for BERT

Results can be found in following output files:
* /tmp/bert_pretrain/phase_2/result_rank_0.txt for TensorFlow BERT
* /tmp/BERT_PRETRAINING/results/checkpoints/result_rank_0.txt for PyTorch BERT

To get the TTT from the training script output, run following command:

```bash
grep 'run_start\|run_stop' /path/to/output/file | grep worker0 | awk '{print $5}' | tr -d ',' | paste -sd " " - | awk '{print ($2 - $1) / 1000 / 60}'
```

As each run uses only about 3% of the MLPerf BERT dataset, the TTT is expected to vary from run to run. See the example below.
All results are presented in minutes as TTT with the number of blocks needed to converge indicated in brackets.

| Test idx | TensorFlow BERT | PyTorch BERT | PyTorch BERT 8 nodes |
| -------- | --------------- | ------------ | -------------------- |
| 1        | 13.031 (15)     | 14.914 (17)  | 2.068 (15)           |
| 2        | 15.631 (18)     | 15.784 (18)  | 2.208 (16)           |
| 3        | 13.901 (16)     | 14.894 (17)  | 1.929 (14)           |
| 4        | 13.896 (16)     | 14.010 (16)  | 2.068 (15)           |
| 5        | 14.780 (17)     | 16.632 (19)  | 2.342 (17)           |
| 6        | 13.026 (15)     | 12.268 (14)  | 2.204 (16)           |
| 7        | 14.761 (17)     | 15.780 (18)  | 2.204 (16)           |
| 8        | 14.763 (17)     | 14.909 (17)  | 2.070 (15)           |
| 9        | 13.025 (15)     | 14.020 (16)  | 2.072 (15)           |
| 10       | 14.766 (17)     | 14.035 (16)  | 1.929 (14)           |

## Training ResNet50

### Training TensorFlow ResNet50

1. Inside the mlperf3.0 TensorFlow container, install Resnet50 requirements.
    ```bash
    export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
    pip install -r $RESNET_IMPLEMENTATIONS/TensorFlow/computer_vision/Resnets/resnet_keras/requirements.txt
    ```

2. Run the training.
    ```bash
    cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-TF
    ./launch_keras_resnet_hvd.sh --config $(pwd)/batch_256.cfg --jpeg-data-dir /root/datasets/imagenet --log_dir /tmp/resnet_log
    ```

### Training PyTorch ResNet50

1. Inside the mlperf3.0 PyTorch container, install Resnet50 requirements.
    ```bash
    export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
    pip install -r $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-PT/PyTorch/requirements.txt
    ```

2. Run the training.
    ```bash
    cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-PT
    ./launch_resnet.sh --config batch_256.cfg --data-dir /root/datasets/imagenet
    ```

### TTT (Time to Train) Calculation for ResNet50

To get the TTT from the training script output, run following command:

```bash
grep 'run_start\|run_stop' /tmp/resnet_log/result_rank_0.txt | grep worker0 | awk '{print $5}' | tr -d ',' | paste -sd " " - | awk '{print ($2 - $1) / 1000 / 60}'
```

According to our experiment, Habana MLPerf ResNet50 can converge in 16.0 minutes with 35 epochs for TensorFlow. PyTorch ResNet50 converges in 16.45 minutes.
See examples below. TTT given in minutes.

| Test idx | TensorFlow ResNet50 | PyTorch ResNet50 |
| -------- | ------------------- | ---------------- |
| 1        | 16.003              | 16.474           |
| 2        | 15.967              | 16.466           |
| 3        | 15.974              | 16.437           |
| 4        | 15.995              | 16.456           |
| 5        | 16.003              | 16.458           |

## Training Unet3D

### Training PyTorch Unet3D

1. Log into mlperf3.0 PyTorch container.

2. Install additional requirements required for UNet3D and execute `run_and_time.sh` script:
    ```
    export UNET3D_PT_DIR=/root/MLPERF/Habana/benchmarks/unet3d/implementations/PyTorch
    cd $UNET3D_PT_DIR
    pip install -r requirements.txt
    DATASET_DIR=$UNET3D_PT_DIR/kits19/preprocessed_data/ bash run_and_time.sh --config config_HLS2_1x8x7.sh
    ```

### TTT (Time to Train) Calculation for PyTorch Unet3D

To get the TTT from the training script output, run the following command:

```bash
grep 'run_start\|run_stop' /tmp/result_rank_0.txt | awk '{print $5}' | tr -d ',' | paste -sd " " - | awk '{print ($2 - $1) / 1000 / 60}'
```

According to our experiment, Habana MLPerf PyTorch Unet3D can converge in 20.52 minutes on average. See examples below. TTT given in minutes.

| Test idx | PyTorch Unet3D |
| -------- | -------------- |
| 1        | 17.978         |
| 2        | 18.959         |
| 3        | 19.446         |
| 4        | 19.692         |
| 5        | 20.185         |
| 6        | 21.411         |
| 7        | 22.640         |
| 8        | 26.069         |


## Training GPT3-175B

All the training steps for GPT3-175B should be performed in mlperf3.0 PyTorch container.

### Installing Requirements

The following requirements need to be installed on all machines participating in the training:
```
pip install git+https://github.com/HabanaAI/DeepSpeed.git
pip install -r /root/MLPERF/Habana/benchmarks/gpt3/requirements.txt
```

### Run and time

Running GPT3 requires multiple machines. For example, 32 HLS2 machines: `HLS-Gaudi2-N32-PT system` or 48 HLS2 machines `HLS-Gaudi2-N48-PT system`.

set the paths for the dataset and the universal checkpoint.
```
export DATASET_DIR=/root/datasets/gpt3/c4/preprocessed_c4_spm
export CHECKPOINT_DIR=/root/datasets/gpt3/universal-checkpoint
```

Please make sure /root/shared/hosts file contains a list of IPs for HLS2 machines, and that SSH communication is properly configured.
For further details, refer to points 5 and 6 [here](#build-and-deploy-habanalabs-mlperf-training-30-container).

#### Running GPT3 on HLS-Gaudi2-N32-PT System
```
bash /root/MLPERF/Habana/benchmarks/gpt3/run_gpt.sh --data-dir $DATASET_DIR/ --universal-ckpt-path $CHECKPOINT_DIR/ \
--hosts /root/shared/hosts --output-dir /root/scratch --num-nodes 32 --data-parallel-size 4 --save-checkpoints false --mllog-output-path /root/scratch/result.txt --train-samples 6782976
```

#### Running GPT3 on HLS-Gaudi2-N48-PT System
```
bash /root/MLPERF/Habana/benchmarks/gpt3/run_gpt.sh --data-dir $DATASET_DIR/ --universal-ckpt-path $CHECKPOINT_DIR/ \
--hosts /root/shared/hosts --output-dir /root/scratch --num-nodes 48 --data-parallel-size 6 --save-checkpoints false --mllog-output-path /root/scratch/result.txt --train-samples 6782976
```

The `--save-checkpoints` is set to `false` as 96l checkpoints take a lot of disc space. In order to save the checkpoint after the run or save it with some frequency, please use `--save-checkpoints true` and manipulate `--save-interval` parameter.
The script will start from universal checkpoint and train up to 416 steps or the time, when validation log perplexity is below 2.69. According to the convergence point of GPT3 on HLS system, it should approximately run for 384 steps in order to reach 2.69 validation log perplexity. To reduce number of steps, you can use `--exit-interval` parameter or reduce train samples by `--train-samples` parameter.

### TTT (Time to Train) Calculation for GPT3-175B

To get the TTT from the training script output, run the following command:

```bash
grep 'run_start\|run_stop' /root/scratch/result.txt | awk '{print $5}' | tr -d ',' | paste -sd " " - | awk '{print ($2 - $1) / 1000 / 60}'
```

See the examples below, where TTT is given in minutes.

| Test idx | N48 GPT3-175B | N32 GPT3-175B |
| -------- | ------------- | ------------- |
| 1        | 311.322       | 442.578       |
| 2        | 325.340       | 460.296       |
| 3        | 311.945       | 441.305       |

## Supported Configurations

| Validated on | SynapseAI Version | Framework Version(s) |   Mode   |
| :----------: | :---------------: | :------------------: | :------: |
|    Gaudi2    |      1.12.1       |  TensorFlow 2.13.0   | Training |
|    Gaudi2    |      1.12.1       |    PyTorch 2.0.1     | Training |

## Changelog
### 1.12.0
- Removed the setting of the PT_HPU_LAZY_MODE environment variable in the script for Bert and ResNet50.
- Removed unused PT_HPU_ENABLE_SYNC_OUTPUT_HOST environment variable.
### 1.11.0
 - Updated scripts to cover MLPerf 3.0 submission.
 - Switched UNet3D, Bert, ResNet50 from HMP to Autocast.
 - Added script for ImageNet unpacking.
 - Reworked scripts and instruction for TensorFlow BERT data preprocessing.
 - Add clearing deepspeed_config to force deepspeed to take config from args.deepspeed_configuration at initialize()
### 1.10.0
 - Updated scripts to cover MLPerf 3.0 submission.
### 1.9.0
 - Disabled auto dynamic shape support for Habana devices for PyTorch ResNet50.
### 1.8.0
- Prepared new scripts for PyTorch BERT data preprocessing.
- Moved data preprocessing instructions to docker environment.
### 1.7.0
 - Updated scripts to cover MLPerf 2.1 submission.
### 1.6.0
- Removed obsolete files from TensorFlow/nlp/bert.
### 1.5.0
- Updated scripts to cover MLPerf 2.0 submission.
- Cleaned up ResNet requirements compared to the originally submitted ones.
- Removed run_bert_docker.sh and run_resnet50_docker.sh scripts.
### 1.4.0
- Switched from the deprecated TF_ENABLE_BF16_CONVERSION to TF_BF16_CONVERSION.
- Added TF_ENABLE_DYNAMIC_SHAPES to MLPerf launchers.
### 1.3.0
- Updated requirements.txt file for BERT and ResNet.
