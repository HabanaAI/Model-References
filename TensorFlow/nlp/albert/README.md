# ALBERT for TensorFlow
This directory provides a script to train a ALBERT models to achieve state-of-the-art accuracy, and is tested and maintained by Habana. To obtain model performance data, refer to the [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model-References](../../../../README.md)
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training and Examples](#training-and-examples)
* [Advanced](#advanced)
* [Supported Configuration](#supported-configuration)
* [Changelog](#changelog)

## Model Overview

ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm by Google. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation.
This release supports Albert Fine-tuning on 1 and 8 cards.

Our implementation is a fork of [Google Research ALBERT](https://github.com/google-research/albert). 

### ALBERT Fine-Tuning
- Suited for tasks:
    - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, in which systems aim to identify if two sentences are paraphrases of each other.
    - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of
       questions posed by crowdworkers on a set of Wikipedia articles, in which the answer to every question is a segment
       of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Default hyperparameters:
    - dataset: squad
    - predict_batch_size: 8
    - max_seq_length: 384
    - doc_stride: 128
    - max_query_length: 64
    - learning_rate: 5e-5
    - num_train_epochs: 2.0
    - warmup_proportion: 0.1
    - save_checkpoints_steps: 5000
    - do_lower_case: true
    - do_train: true
    - do_predict: true
    - use_einsum: false
    - n_best_size: 20
    - max_answer_length: 30
- The output will be saved in $HOME/tmp by default.

## Setup
Please follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References
In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. You can run the
[`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```
Go to the ALBERT directory:

```bash
cd Model-References/TensorFlow/nlp/albert
pip install -r requirements.txt
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/Model-References
```

### Download Datasets
For fine-tuning tasks, since it is using the same datasets as in BERT, please follow the steps in Model-References/TensorFlow/nlp/bert/README.md section "Download and Pre-process the Datasets for Pre-training and Fine-tuning".

## Training and Examples

Download the pre-trained model to the current folder:

- For ALBERT Base:
```bash
$PYTHON download/download_pretrained_model.py https://storage.googleapis.com/albert_models/ albert_base_v1
```

- For ALBERT Large:
```bash
$PYTHON download/download_pretrained_model.py https://storage.googleapis.com/albert_models/ albert_large_v1
```

The training can be run with `run_pretraining.py` for pre-training, `run_classifier` for fine-tuning
with MRPC dataset, and `run_squad_v1.py` with SQuAD dataset:
The following examples assume that the datasets are in a directory /data/tensorflow/.

### Single Card and Multi-Card Training Examples
**Run training on 1 HPU:**
Fine-tuning of ALBERT Large, 1 HPU, in bfloat16 precision using SQuAD dataset:
```bash
TF_BF16_CONVERSION=/path/to/Model-References/TensorFlow/nlp/albert/bf16_config/albert.json \
  $PYTHON run_squad_v1.py \
    --train_feature_file=/root/tmp/albert_large/train_feature_file.tf_record \
    --predict_feature_file=/root/tmp/albert_large/predict_feature_file.tf_record \
    --predict_feature_left_file=/root/tmp/albert_large/predict_feature_left_file.tf_record \
    --spm_model_file=albert_large_v1/30k-clean.model \
    --vocab_file=albert_large_v1/30k-clean.vocab \
    --albert_config_file=albert_large_v1/albert_config.json \
    --init_checkpoint=albert_large_v1/model.ckpt-best \
    --do_train=True \
    --train_file=/data/tensorflow/squad/train-v1.1.json \
    --do_predict=True \
    --predict_file=/data/tensorflow/squad/dev-v1.1.json \
    --train_batch_size=32 \
    --learning_rate=3e-05 \
    --num_train_epochs=2 \
    --max_seq_length=384 \
    --doc_stride=128 \
    --output_dir=/root/tmp/albert_large \
    --use_horovod=false \
    --enable_scoped_allocator=False \
    --save_checkpoints_steps=5000
```
**Run training on 8 HPUs:**

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/TensorFlow/Tensorflow_Scaling_Guide/Horovod_Scaling/index.html#mpirun-configuration).


- Fine-tuning of ALBERT Large, 8 HPUs, in bfloat16 precision using SQuAD dataset:
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/albert/

  mpirun --allow-run-as-root \
    -x TF_BF16_CONVERSION=/path/to/Model-References/TensorFlow/nlp/albert/bf16_config/albert.json \
    --tag-output \
    --merge-stderr-to-stdout \
    --output-filename /root/tmp/albert_log \
    --bind-to core \
    --map-by socket:PE=7 \
    -np 8 \
    $PYTHON run_squad_v1.py \
      --train_feature_file=/root/tmp/albert_large/train_feature_file.tf_record \
      --predict_feature_file=/root/tmp/albert_large/predict_feature_file.tf_record \
      --predict_feature_left_file=/root/tmp/albert_large/predict_feature_left_file.tf_record \
      --spm_model_file=albert_large_v1/30k-clean.model \
      --vocab_file=albert_large_v1/30k-clean.vocab \
      --albert_config_file=albert_large_v1/albert_config.json \
      --init_checkpoint=albert_large_v1/model.ckpt-best \
      --do_train=True \
      --train_file=/data/tensorflow/squad/train-v1.1.json \
      --do_predict=True \
      --predict_file=/data/tensorflow/squad/dev-v1.1.json \
      --train_batch_size=32 \
      --learning_rate=3e-05 \
      --num_train_epochs=2 \
      --max_seq_length=384 \
      --doc_stride=128 \
      --output_dir=/root/tmp/albert_large \
      --use_horovod=true \
      --enable_scoped_allocator=False \
      --save_checkpoints_steps=5000  \
  2>&1 | tee ~/hlogs/albert_large_finetuning_bf16_squad_8_cards.txt
  ```

- Fine-tuning of ALBERT Large, 8 HPUs, in bfloat16 precision using SQuAD dataset on a K8s single server. Make sure to download the pre-trained model for ALBERT Large or Base to a location that can be accessed by the workers, such as the folder for datasets. Make sure the Python packages in `requirements.txt` are installed for all the workers:

  ```bash
  mpirun --allow-run-as-root \
         --bind-to core \
         --map-by socket:PE=6 \
         -np 8 \
         --tag-output \
         --merge-stderr-to-stdout \
         bash -c "cd /root/Model-References/TensorFlow/nlp/albert;\
         $PYTHON run_squad_v1.py \
            --train_feature_file=/root/tmp/albert_large/train_feature_file.tf_record \
            --predict_feature_file=/root/tmp/albert_large/predict_feature_file.tf_record \
            --predict_feature_left_file=/root/tmp/albert_large/predict_feature_left_file.tf_record \
            --spm_model_file=/data/tensorflow/albert_large_v1/30k-clean.model \
            --vocab_file=/data/tensorflow/albert_large_v1/30k-clean.vocab \
            --albert_config_file=/data/tensorflow/albert_large_v1/albert_config.json \
            --init_checkpoint=/data/tensorflow/albert_large_v1/model.ckpt-best \
            --do_train=True \
            --train_file=/data/tensorflow/squad/train-v1.1.json \
            --do_predict=True \
            --predict_file=/data/tensorflow/squad/dev-v1.1.json \
            --train_batch_size=32 \
            --learning_rate=3e-05 \
            --num_train_epochs=2 \
            --max_seq_length=384 \
            --doc_stride=128 \
            --output_dir=/root/tmp/albert_large \
            --use_horovod=true \
            --enable_scoped_allocator=False \
            --save_checkpoints_steps=5000"  \
  2>&1 | tee ~/hlogs/albert_large_ft_squad_8cards.txt
  ```

## Advanced

The following scripts are provided:
* `run_classifier.py`:  Script implementing MRPC task.
* `run_squad_v1.py`:  Script implementing SQUAD task.

## Supported Configuration

| Model | xHPU | Device | SynapseAI Version | TensorFlow Version(s)  |
|:------|:----:|:------:|:-----------------:|:-----:|
|Albert-Large FT (SQUAD) | 1 HPU | Gaudi  | 1.3.0             | 2.8.0 , 2.7.1|
|Albert-Large FT (SQUAD) | 8 HPU | Gaudi  | 1.3.0             | 2.8.0 , 2.7.1 |

## Changelog
### 1.4.0
* Imported horovod-fork package directly instead of using Model-References' TensorFlow.common.horovod_helpers; wrapped horovod import with a try-catch block so that the user is not required to install this library when the model is being run on a single card.
* References to custom demo script were replaced by community entry points in README.
* Switched from deprecated flag TF_ENABLE_BF16_CONVERSION to TF_BF16_CONVERSION.
### 1.3.0
* Added handling of save_checkpoints_steps parameter and changed default to 5000.
* Removed obsolete demo_albert (bash script).
* Moved BF16 config json file from TensorFlow/common/ to model's dir.
* Updated requirements.txt.
* Removed redundant imports.
* Changed `python` or `python3` to `$PYTHON` to execute correct version based on environment setup.
### 1.2.0
* Cleaned up script from deprecated habana_model_runner.
