# Running Habana MLPerf™ Benchmarks

This directory provides instructions to reproduce Habana's results for [MLPerf Training v2.1](https://habana.ai/since-habanas-last-mlperf-submission/) **on 1 server with 8 Gaudi2 cards.**

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/)

MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use is strictly prohibited.

## Table of Contents 
- [Model-References](../../../README.md)
- [Setup](#setup)
- [Training Data for TensorFlow BERT](#training-data-for-tensorflow-bert)
- [Training Data for PyTorch BERT](#training-data-for-pytorch-bert)
- [Training Data for ResNet50](#training-data-for-resnet50)
- [Build and Deploy HabanaLabs MLPerf Training 2.1 Container](#build-and-deploy-habanalabs-mlperf-training-21-container)
- [Training BERT](#training-bert)
- [Training ResNet50](#training-resnet50)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)


## Setup

1. Choose a root directory to store the code and data for MLPerf BERT and ResNet, and make a scratch directory for later use.

```bash
export MLPERF_ROOT=/path/to/mlperf/root
mkdir -p $MLPERF_ROOT
mkdir $MLPERF_ROOT/scratch
```

2. Follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the benchmarks on Gaudi.

### Clone Habana Model-References

Clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

```bash
cd $MLPERF_ROOT
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

**Note:** If the repository is not in the PYTHONPATH, make sure to update by running the below.
```bash
export PYTHONPATH=/path/to/Model-References:$PYTHONPATH
```

### Install Model Requirements

Install the required packages using pip:

```bash
pip install -r $MLPERF_ROOT/Model-References/MLPERF2.1/Habana/benchmarks/requirements.txt
```
## Training Data for TensorFlow BERT

The following resources are required:

* Unpacked dataset in $DATA/bert_pretraining/training
* Packed dataset in $DATA/train/packed_data_500
* Evaluation dataset in $DATA/mlperf_bert_eval_dataset
* Initial checkpoint in $DATA/MLPerf_BERT_checkpoint/model.ckpt-28252
* BERT configuration in $DATA/MLPerf_BERT_checkpoint

To train BERT, follow the steps below. 

1. Create directories to store the TFRecords, the evaluation set, and the results.

```bash
cd $MLPERF_ROOT/Model-References/MLPERF2.1/Habana/benchmarks/bert/pretraining
mkdir tfrecord_dir eval_intermediate mlperf_bert
```

2. Download `vocab.txt` and `bert_config.json` from [Google
   Drive](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT)
   into the current directory.

3. Download the dataset into the current directory and uncompress it. You can
   download `results_text.tar.gz` or `results_text.zip` from [Google
   Drive](https://drive.google.com/corp/drive/u/0/folders/1cywmDnAsrP5-2vsr8GDc6QUc7VWe-M3v).
   This may take some time because it is over 4GB in size. You will now have a
   `results4` directory containing a file named `eval.txt`, and 500 files
   named `part-00###-of-00500`.

4. Execute the following commands to run `create_pretraining_data.py` 500 times.
   This will create TFRecord files `part-00###-of-00500` of size totalling ~365GB.

```bash
export RESULTS=`pwd`/results4
export TFRECORD_DIR=`pwd`/tfrecord_dir
export EVAL=`pwd`/eval_10k
export VOCAB=`pwd`/vocab.txt
export BERT_CONFIG=`pwd`/bert_config.json
for H in {0..4}
do
    for T in {0..9}
    do
        for O in {0..9}
        do
            python3 create_pretraining_data.py --input_file=$RESULTS/part-00${H}${T}${O}-of-00500 \
            --output_file=$TFRECORD_DIR/part-00${H}${T}${O}-of-00500 \
            --vocab_file=$VOCAB \
            --do_lower_case=True \
            --max_seq_length=512 \
            --max_predictions_per_seq=76 \
            --masked_lm_prob=0.15 \
            --random_seed=12345 \
            --dupe_factor=10
        done
    done
done
```

5. The above step can take over 36 hours. However, in parallel, you can create the evaluation set as follows:

```bash
python3 create_pretraining_data.py --input_file=$RESULTS/eval.txt \
    --output_file=eval_intermediate/eval_10k --vocab_file=vocab.txt \
    --do_lower_case=True --max_seq_length=512 --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=10
python3 pick_eval_samples.py --input_tfrecord=eval_intermediate/eval_10k \
    --output_tfrecord=eval_10k --num_examples_to_pick=10000
```

6. Create directories for the packed, unpacked and evaluation datasets, and copy config checkpoint subfolder.

```bash
export DATA=$MLPERF_ROOT/data
mkdir -p $DATA/bert_pretraining $DATA/train/packed_data_500 \
  $DATA/mlperf_bert_eval_dataset $DATA/MLPerf_BERT_checkpoint \
cp $BERT_CONFIG $DATA/MLPerf_BERT_checkpoint/bert_config.json
```

7. Prepare the packed dataset. This will create several files prefixed with `strategy_`.

```bash
python3 scripts/pack_pretraining_data_tfrec.py \
  --input-glob "$TFRECORD_DIR/" \
  --output-dir $DATA/train/packed_data_500 \
  --max-files 500
```

8. Move the unpacked data and the evaluation dataset under the data directory.

```bash
mv $TFRECORD_DIR $DATA/bert_pretraining/training
mv $EVAL $DATA/mlperf_bert_eval_dataset/eval_10k
```

9. Download the checkpoint files from [Google
   Drive](https://drive.google.com/drive/u/0/folders/108tvJyplmFN4Ee5VXzfXUZStuBL4w4PD)
   to the `$DATA/MLPerf_BERT_checkpoint` directory.

## Training Data for PyTorch BERT

### Dataset Preparation

1. Download the dataset and checkpoint, and locate them [here](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT).

2. Set and create PyTorch BERT data folder.
```bash
export PYTORCH_BERT_DATA=$MLPERF_ROOT/data/pytorch_bert
mkdir -p $PYTORCH_BERT_DATA
```

3. Follow the steps in [Download and prepare the data](https://github.com/mlcommons/training_results_v2.0/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch#download-and-prepare-the-data) to download and preprocess the data. Use `$PYTORCH_BERT_DATA` instead of `/workspace/bert_data`. 

At this stage, ```$PYTORCH_BERT_DATA/phase1``` checkpoint and  ```$PYTORCH_BERT_DATA/hdf5/eval_varlength``` evaluation data are ready, while ```$PYTORCH_BERT_DATA/hdf5/training_4320/hdf5_4320_shards_uncompressed``` training data requires packing as described in the following section.

### Training Data Packing

After the training data is ready, pack them using a similar code as described in [GraphCore for v1.0 Submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data). 

```bash
cd $MLPERF_ROOT/Model-References/MLPERF2.1/Habana/benchmarks/bert/implementations/PyTorch
pip3 install -r requirements.txt
python3 pack_pretraining_data_pytorch.py \
    --input_dir=$PYTORCH_BERT_DATA/hdf5/training_4320/hdf5_4320_shards_uncompressed \
    --output_dir=$PYTORCH_BERT_DATA/packed \
    --max_predictions_per_seq=76
```

For further details, refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027). 

## Training Data for ResNet50

1. Follow the instructions under [Stage 1](https://github.com/mlcommons/training/tree/master/image_classification#2-datasetenvironment)
to create TFRecords from ImageNet data.
   * Export ImageNet home directory as IMAGENET_HOME.
   ```bash
   export IMAGENET_HOME=$MLPERF_ROOT/data/imagenet
   ```

2. Apply additional reorganization on jpeg evaluation files.

```bash
cd $IMAGENET_HOME/validation
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```


## Build and Deploy HabanaLabs MLPerf Training 2.1 Container

To build MLPerf training 2.1 container, perform the following:

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
      export CONTAINER_NAME=mlperf2_1
      ```
      * Example on PyTorch Container:  
      ```bash
      # NOTE: The below is only an example value. Replace [SynapseAI version] and [PT version] to match your setup and Supported Configuration. 
      export DOCKER_IMAGE=vault.habana.ai/gaudi-docker/[SynapseAI version]/ubuntu20.04/habanalabs/pytorch-installer-[PT Version]:latest
      export CONTAINER_NAME=mlperf2_1
      ```


3. Create  `mlperf2_1 container` by running the following command. 

```bash
docker run --privileged --security-opt seccomp=unconfined \
  --name $CONTAINER_NAME -td                              \
  -v /dev:/dev                                            \
  --device=/dev:/dev                                      \
  -e LOG_LEVEL_ALL=6                                      \
  -v /sys/kernel/debug:/sys/kernel/debug                  \
  -v /tmp:/tmp                                            \
  -v $MLPERF_ROOT/Model-References/MLPERF2.1:/root/MLPERF \
  -v $MLPERF_ROOT/data:/root/datasets                     \
  -v $MLPERF_ROOT/scratch:/root/scratch                   \
  --cap-add=sys_nice --cap-add=SYS_PTRACE                 \
  --user root --workdir=/root --net=host                  \
  --ulimit memlock=-1:-1 ${DOCKER_IMAGE}
```

4. Start the docker.

```bash
docker exec $CONTAINER_NAME bash -c "service ssh start"
docker exec -it $CONTAINER_NAME bash
```

5. In the docker, create a hosts file that contains a list of hosts in the cluster to /root/shared. The example below is for single node.

```bash
mkdir /root/shared
echo your-machine-ip > /root/shared/hosts
apt update
apt install -y numactl
```

## Training BERT

### Training on TensorFlow BERT

1. Inside the container, install BERT requirements.

```bash
export BERT_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/bert/implementations
pip install -r $BERT_IMPLEMENTATIONS/TensorFlow/nlp/bert/requirements.txt
```

2. Run the training.   
```bash
cd $BERT_IMPLEMENTATIONS/HLS-Gaudi2-TF
./launch_bert_hvd.sh --config defaults.cfg
```

### Training on PyTorch BERT

1. Inside the container, install BERT requirements.

```bash
export BERT_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/bert/implementations
pip install -r $BERT_IMPLEMENTATIONS/PyTorch/requirements.txt
```

2. Run the training. 
```bash
export PYTORCH_BERT_DATA=/root/datasets/pytorch_bert
cd /root/MLPERF/Habana/benchmarks/bert/implementations/HLS-Gaudi2-PT
./launch_bert_pytorch.sh --input-dir $PYTORCH_BERT_DATA/packed
--phase1-ckpt $PYTORCH_BERT_DATA/phase1/model.ckpt-28252.pt --eval-dir $PYTORCH_BERT_DATA/hdf5/eval_varlength
```

### TTT (Time to Train) Calculation for BERT

All results can be found at /root/scratch/bert inside the container. 

To get the TTT from the training script output, run following command: 

```bash
grep 'run_start\|run_stop' /root/scratch/bert/result_rank_0.txt |grep worker0|awk '{print $5}' | tr -d ',' | paste -sd " " - | awk '{print ($2 - $1) / 1000 / 60}'
```

As each run uses only about 3% of the MLPerf BERT dataset, the TTT is expected to vary from run to run. See the example below. 

| Test idx | # blocks to converge | TTT (minutes) |
| -------- | -------------------- | ------------- |
| 1        | 17                   | 16.2659       |
| 2        | 16                   | 15.3234       |
| 3        | 15                   | 14.3685       |
| 4        | 18                   | 17.2271       |
| 5        | 17                   | 16.2732       |

## Training ResNet50

### Training on TensorFlow ResNet50 

1. Inside the container, install Resnet50 requirements.

```bash
export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
pip install -r $RESNET_IMPLEMENTATIONS/TensorFlow/computer_vision/Resnets/resnet_keras/requirements.txt
```

2. Run the training. 
```bash
cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-TF
./launch_keras_resnet_hvd.sh --config batch_256.cfg --cpu-pin cpu --jpeg-data-dir /root/datasets/imagenet
```

### Training on PyTorch ResNet50 

1. Inside the container, install Resnet50 requirements.

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

Run the following command to get the TTT from the training script output `launch_keras_resnet_hvd.sh or launch_resnet.sh invocation`, assuming it is captured in a file:

```bash
grep 'run_start\|run_stop' /path/to/captured/output/file |grep worker0|awk '{print $5}' | tr -d ',' | paste -sd " " - | awk '{print ($2 - $1) / 1000 / 60}'
```

According to our experiment, Habana MLP ResNet50 can converge in 16.6 mins with 35 epochs. See the example below.

| Test idx | TTT (minutes) |
| -------- | ------------- |
| 1        | 16.5918       |
| 2        | 16.5995       |
| 3        | 16.6000       |
| 4        | 16.5848       |
| 5        | 16.5990       |

## Supported Configurations

| Device | SynapseAI Version | Framework Version(s)  |
|:------:|:-----------------:|:---------------------:|
| Gaudi2 | 1.7.1             | TensorFlow 2.10.1     |
| Gaudi2 | 1.7.1             | TensorFlow 2.8.4      |
| Gaudi2 | 1.7.1             | PyTorch 1.13.0        |

## Changelog
### 1.7.0
Updated scripts to cover MLPerf 2.1 submission.
### 1.6.0
Removed obsolete files from TensorFlow/nlp/bert
### 1.5.0
- Updated scripts to cover MLPerf 2.0 submission.
- cleaned up ResNet requirements compared to the originally submitted ones.
- Removed run_bert_docker.sh and run_resnet50_docker.sh scripts.
### 1.4.0
- Switched from the deprecated TF_ENABLE_BF16_CONVERSION to TF_BF16_CONVERSION.
- Added TF_ENABLE_DYNAMIC_SHAPES to MLPerf launchers. 
### 1.3.0
Updated requirements.txt file for BERT and ResNet. 

