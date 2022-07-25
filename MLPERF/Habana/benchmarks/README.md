# Running Habana MLPerf™ Benchmarks
- [Running Habana MLPerf™ Benchmarks](#running-habana-mlperf-benchmarks)
  - [Setup](#setup)
  - [Training Data for BERT](#training-data-for-bert)
  - [Training Data for ResNet 50](#training-data-for-resnet-50)
  - [Training BERT](#training-bert)
    - [TTT (Time to Train) Calculation for BERT](#ttt-time-to-train-calculation-for-bert)
  - [Training ResNet 50](#training-resnet-50)
    - [TTT (Time to Train) Calculation for ResNet 50](#ttt-time-to-train-calculation-for-resnet-50)
  - [Changelog](#changelog)

This directory contains instructions for reproducing Habana's results for
[MLPerf training
v2.0](https://habana.ai/since-habanas-last-mlperf-submission/)
**on a server with 8 Gaudi cards.**

For more information about training deep learning models on Gaudi, visit
[developer.habana.ai](https://developer.habana.ai/resources/).

MLPerf™ is a trademark and service mark of MLCommons Association in the United
States and other countries. All rights reserved. Unauthorized use strictly
prohibited.

## Setup

1. Choose a root directory where code and data for MLPerf BERT and Resnet will be stored.
   Also make a scratch directory for later use.
```bash
export MLPERF_ROOT=/path/to/mlperf/root
mkdir -p $MLPERF_ROOT
mkdir $MLPERF_ROOT/scratch
```

2. Follow the instructions given in the following link for setting up the
   environment including the `$PYTHON` environment variable: [Gaudi Installation
   Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
   This guide will walk you through the process of setting up your system to run
   the benchmarks on Gaudi.

3. Clone this repository and switch to the branch that matches your SynapseAI
   version. (Run the
   [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
   utility to determine the SynapseAI version.)
```bash
cd $MLPERF_ROOT
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

4. Install the following dependencies. (Installation in a Python virtual environment is recommended.)
```bash
pip install -r $MLPERF_ROOT/Model-References/MLPERF/Habana/benchmarks/requirements.txt
```

## Training Data for BERT

To train BERT, the following resources are required:

* Unpacked dataset in $DATA/bert_pretraining/training
* Packed dataset in $DATA/train/packed_data_500
* Evaluation dataset in $DATA/mlperf_bert_eval_dataset
* Initial checkpoint in $DATA/MLPerf_BERT_checkpoint/model.ckpt-28252
* BERT configuration in $DATA/MLPerf_BERT_checkpoint

1. Create directories to store the TFRecords, the evaluation set, and the results

```bash
cd $MLPERF_ROOT/Model-References/MLPERF/Habana/benchmarks/bert/pretraining
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
   After that, you will have TFRecord files `part-00###-of-00500` of size
   totalling ~365GB.

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

5. The above step can take over 36 hours. However, in parallel, you can create
   the evaluation set as follows:

```bash
python3 create_pretraining_data.py --input_file=$RESULTS/eval.txt \
    --output_file=eval_intermediate/eval_10k --vocab_file=vocab.txt \
    --do_lower_case=True --max_seq_length=512 --max_predictions_per_seq=76 \
    --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=10
python3 pick_eval_samples.py --input_tfrecord=eval_intermediate/eval_10k \
    --output_tfrecord=eval_10k --num_examples_to_pick=10000
```

6. Now make directories for the packed, unpacked and evaluation datasets. Copy config checkpoint subfolder.

```bash
export DATA=$MLPERF_ROOT/data
mkdir -p $DATA/bert_pretraining $DATA/train/packed_data_500 \
  $DATA/mlperf_bert_eval_dataset $DATA/MLPerf_BERT_checkpoint \
cp $BERT_CONFIG $DATA/MLPerf_BERT_checkpoint/bert_config.json
```

7. Prepare the packed dataset. After this, you will have several files
   prefixed with `strategy_`.

```bash
python3 scripts/pack_pretraining_data_tfrec.py \
  --input-glob "$TFRECORD_DIR/" \
  --output-dir $DATA/train/packed_data_500 \
  --max-files 500
```

8. Move the unpacked data and the evaluation dataset under the data directory:

```bash
mv $TFRECORD_DIR $DATA/bert_pretraining/training
mv $EVAL $DATA/mlperf_bert_eval_dataset/eval_10k
```

9. Download the checkpoint files from [Google
   Drive](https://drive.google.com/drive/u/0/folders/108tvJyplmFN4Ee5VXzfXUZStuBL4w4PD)
   to the `$DATA/MLPerf_BERT_checkpoint` directory.

## Training Data for ResNet 50

1. Follow the instructions under Stage 1
[here](https://github.com/mlcommons/training/tree/master/image_classification#2-datasetenvironment)
to create TFRecords from ImageNet data.
   * Use following setting for IMAGENET_HOME
   ```bash
   export IMAGENET_HOME=$MLPERF_ROOT/data/imagenet
   ```

2. Do additional reorganization on jpeg evaluation files

```bash
cd $IMAGENET_HOME/validation
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

## Build and deploy HabanaLabs MLPERF training 2.0 container

Build MLPERF training 2.0 container by

1. Copying ssh keys to enable passwordless ssh to /root/.ssh/
2. Seting enviroment variables for docker command.
   * To a find docker image go to https://vault.habana.ai/ui/repos/tree/General/gaudi-docker
   * Open gaudi-docker directory and select folder matching SynapseAI version (determined by running [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options))
   * Navigate to subdirectories, choosing system and tensorflow versions
   * Choose the docker build version at the end. Most often 'latest' will be used.
   * Navigate to "docker info" tab and note "title" string.
   * Set DOCKER_IMAGE to "title" string with "vault.habana.ai/gaudi-docker/" prefix
   * NOTE: below DOCKER_IMAGE value is just an example.
```bash
# NOTE: example value
export DOCKER_IMAGE=vault.habana.ai/gaudi-docker/1.5.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.9.1:latest
export CONTAINER_NAME=mlperf2_0
```

3. Run following command to create mlperf2_0 container

```bash
sudo docker run --privileged --security-opt seccomp=unconfined \
  --name $CONTAINER_NAME -td                           \
  --device=/dev/hl_controlD0:/dev/hl_controlD0         \
  --device=/dev/hl_controlD1:/dev/hl_controlD1         \
  --device=/dev/hl_controlD2:/dev/hl_controlD2         \
  --device=/dev/hl_controlD3:/dev/hl_controlD3         \
  --device=/dev/hl_controlD4:/dev/hl_controlD4         \
  --device=/dev/hl_controlD5:/dev/hl_controlD5         \
  --device=/dev/hl_controlD6:/dev/hl_controlD6         \
  --device=/dev/hl_controlD7:/dev/hl_controlD7         \
  --device=/dev/hl0:/dev/hl0                           \
  --device=/dev/hl1:/dev/hl1                           \
  --device=/dev/hl2:/dev/hl2                           \
  --device=/dev/hl3:/dev/hl3                           \
  --device=/dev/hl4:/dev/hl4                           \
  --device=/dev/hl5:/dev/hl5                           \
  --device=/dev/hl6:/dev/hl6                           \
  --device=/dev/hl7:/dev/hl7                           \
  -e DISPLAY=$DISPLAY                                  \
  -e LOG_LEVEL_ALL=6                                   \
  -v /sys/kernel/debug:/sys/kernel/debug               \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro                  \
  -v /tmp:/tmp                                         \
  -v $MLPERF_ROOT/Model-References/MLPERF:/root/MLPERF \
  -v $MLPERF_ROOT/data:/root/datasets                  \
  -v $MLPERF_ROOT/scratch:/root/scratch                \
  --cap-add=sys_nice --cap-add=SYS_PTRACE              \
  --user root --workdir=/root --net=host               \
  --ulimit memlock=-1:-1 ${DOCKER_IMAGE}
```

5. Starting the docker

```bash
docker exec $CONTAINER_NAME bash -c "service ssh start"
docker exec -it $CONTAINER_NAME bash
```

5. In docker, creating a hosts file that contains a list of hosts in the cluster to /root/shared
   (example below is for single node):

```bash
mkdir /root/shared
echo your-machine-ip > /root/shared/hosts
apt update
apt install -y numactl
```

## Training BERT

1. Inside the container, install BERT requirements

```bash
export BERT_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/bert/implementations
pip install -r $BERT_IMPLEMENTATIONS/TensorFlow/nlp/bert/requirements.txt
```

2. Run training on
  * AWS DL1 instance
  ```bash
  cd $BERT_IMPLEMENTATIONS/HLS-1-N1
  ./launch_bert_hvd.sh -hls OCP1
  ```

  * HLS1
  ```bash
  cd $BERT_IMPLEMENTATIONS/HLS-1-N1
  ./launch_bert_hvd.sh -hls HLS1
  ```

  * HLS2
  ```bash
  cd $BERT_IMPLEMENTATIONS/HLS-Gaudi2
  ./launch_bert_hvd.sh -hls HLS2
  ```

### TTT (Time to Train) Calculation for BERT

All results can be found at /root/scratch/bert inside container.

The following command can help you get the TTT from the result of the training
script:

```bash
grep 'run_start\|run_stop' /root/scratch/bert/result_rank_0.txt |grep worker0|awk '{print $5}' | tr -d ',' | paste -sd " " - | awk '{print ($2 - $1) / 1000 / 60}'
```

As each run only uses about 3% of the MLPerf bert dataset, the TTT is expected
to vary from run to run. For example:

| Test idx | # blocks to converge | TTT (minutes) |
| -------- | -------------------- | ------------- |
| 1        | 16                   | 75.547        |
| 2        | 17                   | 80.295        |
| 3        | 18                   | 84.808        |
| 4        | 19                   | 89.541        |
| 5        | 16                   | 75.494        |

## Training ResNet 50

1. Inside the container, install Resnet requirements

```bash
export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
pip install -r $RESNET_IMPLEMENTATIONS/TensorFlow/computer_vision/Resnets/resnet_keras/requirements.txt
```

2. Run training on
  * AWS DL1 instance
  ```bash
  cd $RESNET_IMPLEMENTATIONS/HLS-1-N1
  NUM_WORKERS_PER_HLS=8 HLS_TYPE=OCP1 ./launch_keras_resnet_hvd.sh --config batch_256.cfg  --cpu-pin cpu
  ```

  * HLS1
  ```bash
  cd $RESNET_IMPLEMENTATIONS/HLS-1-N1
  NUM_WORKERS_PER_HLS=8 HLS_TYPE=HLS1 ./launch_keras_resnet_hvd.sh --config batch_256.cfg  --cpu-pin cpu
  ```

  * HLS2
  ```bash
  cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2
  NUM_WORKERS_PER_HLS=8 HLS_TYPE=HLS2 ./launch_keras_resnet_hvd.sh --config batch_256.cfg --cpu-pin cpu --jpeg-data-dir /root/datasets/imagenet
  ```

### TTT (Time to Train) Calculation for ResNet 50

The following command can help you get the TTT from the output of the training
script (launch_keras_resnet_hvd.sh invocation), assuming it is captured in a file:

```bash
grep 'run_start\|run_stop' /path/to/captured/output/file |grep worker0|awk '{print $5}' | tr -d ',' | paste -sd " " - | awk '{print ($2 - $1) / 1000 / 60}'
```

According to our experiment, Habana MLP Resnet50 can converge in 63 mins with 38 epochs. For example:

| Test idx | TTT (minutes) |
| -------- | ------------- |
| 1        | 63.0145       |
| 2        | 63.1641       |
| 3        | 63.0369       |

## Supported Configuration

| Device | SynapseAI Version | TensorFlow Version(s)  |
|:------:|:-----------------:|:-----:|
| Gaudi  | 1.5.0             | 2.9.1 |
| Gaudi  | 1.5.0             | 2.8.2 |
| Gaudi2  | 1.5.0             | 2.9.1 |
| Gaudi2  | 1.5.0             | 2.8.2 |

## Changelog
### v1.5.0
- Scripts updated to cover MLPerf 2.0 submission
- Resnet requirements cleaned up compared to originally submitted ones
- Removed run_bert_docker.sh and run_resnet50_docker.sh scripts
### v1.4.0
- Switched from deprecated TF_ENABLE_BF16_CONVERSION to TF_BF16_CONVERSION
- Add TF_ENABLE_DYNAMIC_SHAPES to MLPerf launchers
### v1.3.0
- requirements.txt updated for Bert and ResNet
