# ALBERT

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
* [Model Overview](#model-overview)
* [Setup](#setup)
* [ALBERT Pre-Training](#albert-pre-training)
* [ALBERT Fine-Tuning](#albert-fine-tuning)
* [Training the Model](#training-the-model)
* [Examples](#examples)
* [Preview of TensorFlow ALBERT Python scripts with yaml configuration of parameters](#preview-of-tensorflow-albert-python-scripts-with-yaml-configuration-of-parameters)

## Model Overview

ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm by Google. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation.

Our implementation is a fork of [Google Research ALBERT](https://github.com/google-research/albert).

## Setup
### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the driver.

<br />

### Install container runtime
<details>
<summary>Ubuntu distributions</summary>

### Setup package fetching
1. Download and install the public key:  
```
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add -
```
2. Create an apt source file /etc/apt/sources.list.d/artifactory.list.
3. Add the following content to the file:
```
deb https://vault.habana.ai/artifactory/debian focal main
```
4. Update Debian cache:  
```
sudo dpkg --configure -a
sudo apt-get update
```  
### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo apt install -y habanalabs-container-runtime=0.14.0-420
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>

<details>
<summary>CentOS distributions</summary>

### Setup package fetching
1. Create /etc/yum.repos.d/Habana-Vault.repo.
2. Add the following content to the file:
```
[vault]

name=Habana Vault

baseurl=https://vault.habana.ai/artifactory/centos7

enabled=1

gpgcheck=0

gpgkey=https://vault.habana.ai/artifactory/centos7/repodata/repomod.xml.key

repo_gpgcheck=0
```
3. Update YUM cache by running the following command:
```
sudo yum makecache
```
4. Verify correct binding by running the following command:
```
yum search habana
```
This will search for and list all packages with the word Habana.

### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo yum install habanalabs-container-runtime-0.14.0-420* -y
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>

<details>
<summary>Amazon linux distributions</summary>

### Setup package fetching
1. Create /etc/yum.repos.d/Habana-Vault.repo.
2. Add the following content to the file:
```
[vault]

name=Habana Vault

baseurl=https://vault.habana.ai/artifactory/AmazonLinux2

enabled=1

gpgcheck=0

gpgkey=https://vault.habana.ai/artifactory/AmazonLinux2/repodata/repomod.xml.key

repo_gpgcheck=0
```
3. Update YUM cache by running the following command:
```
sudo yum makecache
```
4. Verify correct binding by running the following command:
```
yum search habana
```
This will search for and list all packages with the word Habana.

### Install habana-container-runtime:
Install the `habana-container-runtime` package:
```
sudo yum install habanalabs-container-runtime-0.14.0-420* -y
```
### Docker Engine setup

To register the `habana` runtime, use the method below that is best suited to your environment.
You might need to merge the new argument with your existing configuration.

#### Daemon configuration file
```bash
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "habana": {
            "path": "/usr/bin/habana-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

You can optionally reconfigure the default runtime by adding the following to `/etc/docker/daemon.json`:
```
"default-runtime": "habana"
```
</details>
<br />

Get the Habana TensorFlow docker image for your OS. The example below is for ubuntu20.04:
```bash
docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

Run the docker container:
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug --net=host --workdir=/root vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/tensorflow-installer-tf-cpu-2.2.2:0.14.0-420
```

## ALBERT Pre-Training
- Suited for datasets:
    - `overfit`
    - provided by user
- Default hyperparameters:
    - dataset: overfit
    - eval_batch_size: 8
    - max_seq_length: 128
    - optimizer: lamb
    - learning_rate: 0.000176
    - num_train_steps: 200
    - num_warmup_steps: 10
    - save_checkpoints_steps: 5000
    - lower_case: true
    - do_train: true
    - do_eval: true
- The output will be saved in $HOME/tmp by default.

## ALBERT Fine-Tuning
- Suited for tasks:
    - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of
       questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment
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

## Training the Model
In the docker container:
```bash
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/TensorFlow/nlp/albert
```

If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/Model-References
```

To see the full list of available options and the descriptions, run:
```bash
./demo_albert -h  # for help
```

## Examples
The following table provides command lines to run the training under various configurations.

| Model        | Task & Dataset        | # Gaudi cards, Batch Size, Seq Length, Precision | Command Line |
|:-------------|-----------------------|:------------------------------------------------:|:--------------------------------------------------|
| ALBERT Base  | Fine-Tuning for MRPC  | 1-card, BS=64, Seq=128, bf16                     | `./demo_albert finetuning -d bf16 -m base -t mrpc -b 64 -s 128`      |
| ALBERT Base  | Fine-Tuning for SQuAD | 1-card, BS=64, Seq=128, bf16                     | `./demo_albert finetuning -d bf16 -m base -t squad -b 64 -s 128`     |
| ALBERT Base  | Fine-Tuning for SQuAD | 8-cards, BS=64, Seq=128, bf16                    | `./demo_albert finetuning -d bf16 -m base -t squad -b 64 -s 128 -v`  |
| ALBERT Large | Fine-Tuning for MRPC  | 1-card, BS=32, Seq=128, bf16                     | `./demo_albert finetuning -d bf16 -m large -t mrpc -b 32 -s 128`     |
| ALBERT Large | Fine-Tuning for SQuAD | 1-card, BS=32, Seq=128, bf16                     | `./demo_albert finetuning -d bf16 -m large -t squad -b 32 -s 128`    |
| ALBERT Large | Fine-Tuning for SQuAD | 8-cards, BS=32, Seq=128, bf16                    | `./demo_albert finetuning -d bf16 -m large -t squad -b 32 -s 128 -v` |

## Preview of TensorFlow ALBERT Python scripts with yaml configuration of parameters
There are two yaml configuration files under *hb_configs* directory, **albert_base_default.yaml** and **albert_large_default.yaml**, that specify the  model variant (base or large), subcommand (finetuning or pretraining), dataset (mrpc, squad, or overfit), all of the hyperparameters associated with the model as well as environment variables. Note that these yaml files include the best known configurations for our hardware.

The model can be invoked via `habana_model_runner.py`.
```bash
python3 ../../habana_model_runner.py --model albert --hb_config path_to_yaml_config_file
```

The script automatically downloads the pre-trained model from https://storage.googleapis.com/albert_models/ the first time it is run in the docker container, as well as the dataset, if needed.

To enable multinode training, make sure to update the parameters in the yaml config's 'parameters' section:
```
use_horovod: True

num_workers_per_hls: 8
```
