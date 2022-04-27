# Running Habana MLPerf™ Benchmarks
- [Running Habana MLPerf™ Benchmarks](#running-habana-mlperf-benchmarks)
  - [Table of Contents](#table-of-contents)
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
v1.1](https://habana.ai/mlperf-ai-training-benchmark-habana-gaudi-performance-and-scale-results/)
**on a server with 8 Gaudi cards.**

For more information about training deep learning models on Gaudi, visit
[developer.habana.ai](https://developer.habana.ai/resources/).

MLPerf™ is a trademark and service mark of MLCommons Association in the United
States and other countries. All rights reserved. Unauthorized use strictly
prohibited.

## Table of Contents
  * [Model-References](../../../README.md)
  * [Setup](#setup)
    * [Training Data for BERT](#training-data-for-bert)
    * [Training Data for ResNet 50](#training-data-for-resnet-50)
  * [Training BERT](#training-bert)
    * [TTT (Time to Train) Calculation for BERT](#ttt-time-to-train-calculation-for-bertt)
  * [Training ResNet 50](#training-resnet-50)
    * [TTT (Time to Train) Calculation for ResNet 50](#ttt-time-to-train-calculation-for-resnet-50)

## Setup

1. Follow the instructions given in the following link for setting up the
   environment including the `$PYTHON` environment variable: [Gaudi Installation
   Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html).
   This guide will walk you through the process of setting up your system to run
   the benchmarks on Gaudi.

2. Clone this repository and switch to the branch that matches your SynapseAI
   version. (Run the
   [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
   utility to determine the SynapseAI version.)
```bash
export ROOT=`pwd`
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

3. Install the following dependencies. (Installation in a Python virtual environment is recommended.)
```bash
pip install -r $ROOT/Model-References/MLPERF/Habana/benchmarks/requirements.txt
```
### Training Data for BERT

To train BERT, the following resources are required:

* Unpacked dataset in $DATA/bert_pretraining/training
* Packed dataset in $DATA/train/packed_data_500
* Evaluation dataset in $DATA/mlperf_bert_eval_dataset
* Initial checkpoint in $DATA/MLPerf_BERT_checkpoint/model.ckpt-28252
* BERT configuration in $DATA/MLPerf_BERT_checkpoint

1. Create directories to store the TFRecords, the evaluation set, and the results

```bash
cd Model-References/MLPERF/Habana/benchmarks/bert/pretraining
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

6. Now make a directory for the packed dataset and the results:

```bash
export DATA=`pwd`/data
mkdir -p $DATA/bert_pretraining $DATA/train/packed_data_500 \
  $DATA/mlperf_bert_eval_dataset $DATA/MLPerf_BERT_checkpoint \
  mlperf_result
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

Create a directory where the results can be stored.
```bash
cd Model-References/MLPERF/Habana/benchmarks/resnet
mkdir resnet50_result
```

Follow the instructions under Stage 1
[here](https://github.com/mlcommons/training/tree/master/image_classification#2-datasetenvironment)
to create TFRecords from ImageNet data.

## Training BERT

1. Modify `run_bert_docker.sh` (in this repository) so that
* `RESULTS_DIR` refers to the `mlperf_result` directory you previously created,
* `DATASET_DIR` refers to the `data` directory where the packed, upacked, and evaluation datasets are stored, and
* `CODE_DIR` refers to the `MLPERF` directory in this repository.
These directories will be mounted inside the running
docker container.

2. Execute `./run_bert_docker.sh` to pull (if necessary), start, and attach to the Docker container.

3. Inside the container,
```bash
echo 127.0.0.1 > /root/shared/hosts
pip install -r /root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert/requirements.txt
apt update
apt install -y numactl
cd /root/MLPERF/Habana/benchmarks/bert/implementations/HLS-1-N1
```
If running on AWS DL1 instance,
```bash
./launch_bert_hvd.sh
```

If running elsewhere,
```bash
./launch_bert_hvd.sh -hls HLS1
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

1. Modify `run_resnet50_docker.sh` (in this repository) so that
* `MLPERF_HOST` refers to the `MLPERF` directory in this repository.
* `IMAGENET_HOST` refers to the TFRecords directory
* `RESULTS_DIR` refers to the `resnet50_result` directory created previously.

2. If necessary, modify the last line of `./run_resnet50_docker.sh` based on the comments given in the script.

3. Execute `./run_resnet50_docker.sh` to pull (if necessary) the Docker container and
   start the training inside it.

### TTT (Time to Train) Calculation for ResNet 50

The following command can help you get the TTT from the output of the training
script, assuming it is captured in a file:

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
| Gaudi  | 1.4.0             | 2.8.0 |
| Gaudi  | 1.4.0             | 2.7.1 |

## Changelog
### v1.3.0
- requirements.txt updated for Bert and ResNet
### v1.4.0
- Switched from deprecated TF_ENABLE_BF16_CONVERSION to TF_BF16_CONVERSION
- Add TF_ENABLE_DYNAMIC_SHAPES to MLPerf launchers