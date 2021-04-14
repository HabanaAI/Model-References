# BERT for PyTorch

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

This folder contains scripts to pre-train and fine-tune BERT model on Habana Gaudi<sup>TM</sup> device to achieve state-of-the-art accuracy.

The BERT demos included in this release are as follows:
- BERT Base fine-tuning for FP32 with MRPC & SQuAD dataset in Eager mode.
- BERT Large fine-tuning for FP32 and BF16 Mixed precision for SQuADv1.1 dataset in Graph mode.
- BERT Large fine-tuning with MRPC dataset for FP32 in Graph mode.
- BERT Large pre-training for FP32 and BF16 Mixed precision for Wikipedia BookCorpus and Wiki dataset in Graph mode.
- Multi node (1xHLS = 8 cards) demo for BERT-Large Fine tuning with FP32 and BF16 Mixed precision in Graph mode.
- Multi node (1xHLS = 8 cards) demo for BERT Large Pretraining with FP32 and BF16 Mixed precision in Graph mode.

Graph mode is supported using torch.jit.trace with check_trace=False.

Each demo script is a wrapper for respective python training scripts. Demo scripts set parameters using environment variables in order to achieve optimum results for each workload. Please refer to demo script for environment variables used for the workload.

## Model Overview
Bidirectional Encoder Representations from Transformers (BERT) is a technique for natural language processing (NLP) pre-training developed by Google.
The original English-language BERT model comes with two pre-trained general types: (1) the BERTBASE model, a 12-layer, 768-hidden, 12-heads, 110M parameter neural network architecture, and (2) the BERTLARGE model, a 24-layer, 1024-hidden, 16-heads, 340M parameter neural network architecture; both of which were trained on the BooksCorpus with 800M words, and a version of the English Wikipedia with 2,500M words.
The Pretraining modeling scripts are derived from a clone of https://github.com/NVIDIA/DeepLearningExamples.git and the fine tuning is based on https://github.com/huggingface/transformers.git.

## Docker Setup
PyTorch BERT is tested with Habana PyTorch docker container 0.14.0-420 for Ubuntu 20.04 and some dependencies contained within it.

### Requirements
* Docker version 19.03.12 or newer
* Sudo access to install required drivers/firmware
  OR
  If they are already installed, skip to #Pre-training or #Fine Tuning

### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the drivers.

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

# Pre-Training
- Located in: `Model-References/PyTorch/nlp/bert/pretraining`
- Suited for datasets:
  - `wiki`, `bookswiki`(combination of BooksCorpus and Wiki datasets)
- Uses optimizer: **LAMB** ("Layer-wise Adaptive Moments optimizer for Batch training").
- Consists of 2 phases:
  - Task 1 - **Masked Language Model** - where given a sentence, a randomly chosen word is guessed.
  - Task 2 - **Next Sentence Prediction** - where the model guesses whether sentence B comes after sentence A
- The resulting (trained) model weights are language-specific (here: english) and has to be further "fitted" to do a specific task (with finetuning).
- Heavy-weight: the training takes several hours or days.

BERT training script supports pre-training of  dataset on BERT large for both FP32 and BF16 mixed precision data type using **Graph mode**.

## Model Setup
The base training and modeling scripts for pretraining are based on a clone of
https://github.com/NVIDIA/DeepLearningExamples.
Please follow the docker and Model-References git clone setup steps.

1. Stop running dockers
```
docker stop $(docker ps -a -q)
```
2. Pull docker image

You can pull Habana PyTorch docker container for either Ubuntu 18.04 or 20.04 as follows:

| Ubuntu Version | Command                                              |
|:---------------|:------------------------------------------------------------------------|
| Ubuntu 18.04 | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/pytorch-installer:0.14.0-420` |
| Ubuntu 20.04 | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420` |

3. Run docker
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420
```
4. In the docker container, clone the repository and go to PyTorch BERT directory:
 ```
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/nlp/bert
 ```
5. Add `Model-References/PyTorch/nlp/bert` folder to PYTHONPATH
```bash
export PYTHONPATH=<path_to_model_references>/Model-References/PyTorch/nlp/bert:$PYTHONPATH
```

### Set up dataset
`Model-References/PyTorch/nlp/bert/pretraining/data` provides scripts to download, extract and preprocess [Wikipedia](https://dumps.wikimedia.org/) and [BookCorpus](http://yknzhu.wixsite.com/mbweb) datasets.
Install the required Python packages in the container:
```
 pip install -r Model-References/PyTorch/nlp/bert/pretraining/requirements.txt
```
Then, go to `data` folder and run the data preparation script.
```
cd Model-References/PyTorch/nlp/bert/pretraining/data
```
So it is recommended to download wiki data set alone using the following command.
```
bash create_datasets_from_start.sh
```
Wiki and BookCorpus data sets can be downloaded by runnining the script as follows.
```
bash create_datasets_from_start.sh wiki_books
```
Note that the pretraining dataset is huge and takes several hours to download. BookCorpus may have access and download constraints. The final accuracy may vary depending on the dataset and its size.
The script creates formatted dataset for the phase1 and 2 of pre-training.

## Training the Model
Clone the Model-References git.
Set up the data set as mentioned in the section "Set up dataset".

```
cd Model-References/PyTorch/nlp/bert
```
Run `./demo_bert -h` for command-line options

i. graph mode, bf16 mixed precision, BS64 for phase1 and BS8 for phase2:
```
./demo_bert --sub-command pretraining --data-type bf16 -b 64 8 --data-dir <dataset path for phase1>  <dataset path for phase2>
```
ii. graph mode, fp32 precision, BS32 for phase1 and BS4 for phase2:
```
./demo_bert --sub-command pretraining --data-type fp32 -b 32 4 --data-dir <dataset path for phase1>  <dataset path for phase2>
```
where, assuming wiki dataset was downloaded,
```
<dataset path for phase1> = pretraining/data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/
<dataset path for phase2> = pretraining/data/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/
```

## Multinode Training
Follow the relevant steps under "Training the Model".
To run multi-node demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-node demo.
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420
```

Number of nodes can be configured using -w option in the demo script.
Use the following commands to run multinode training on 8 cards:

i. graph mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 8 for phase2:
```
./demo_bert_dist --sub-command pretraining --data-type bf16 -b 64 8 -w 8 --data-dir <dataset path for phase1> <dataset path for phase2>
```
ii. graph mode, fp32 precision, per chip batch size of 32 for phase1 and 4 for phase2:
```
./demo_bert_dist --sub-command pretraining --data-type fp32 -b 32 4 -w 8 --data-dir <dataset path for phase1> <dataset path for phase2>
```

## Known Issues

# Fine Tuning
- Located in: `Model-References/PyTorch/nlp/bert/finetuning`
- Suited for tasks:
  - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
  - `squad`: Stanford Question Answering Dataset (**SQuAD**) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
- Uses optimizer: **AdamW** ("ADAM with Weight Decay Regularization").
- Based on model weights trained with pretraining.
- Light-weight: the training takes a minute or so.
- Datasets for MRPC and SQuAD will be automatically downloaded the first time the model is run in the docker container.

The BERT demo uses training scripts and models from https://github.com/huggingface/transformers.git (tag v3.0.2)

## Model Set Up
The training script fine-tunes BERT base and large model on the [Microsoft Research Paraphrase Corpus](https://gluebenchmark.com/) (MRPC) and [Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (SQuADv1.1) dataset.

1. Stop running dockers
```
docker stop $(docker ps -a -q)
```
2. Pull docker image

You can pull Habana PyTorch docker container for either Ubuntu 18.04 or 20.04 as follows:

| Ubuntu Version | Command                                              |
|:---------------|:------------------------------------------------------------------------|
| Ubuntu 18.04 | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu18.04/habanalabs/pytorch-installer:0.14.0-420` |
| Ubuntu 20.04 | `docker pull vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420` |

3. Run docker
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420
```
4. In the docker container, clone the repository and go to PyTorch BERT directory:
 ```
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/nlp/bert
 ```
5. Run `./demo_bert -h` for command-line options

### Set up dataset

#### MRPC dataset preparation:
MRPC dataset can be downloaded using download_glue_data.py script. Download python script from the following link and run the script to download data into `glue_data` directory.
```
mkdir -p $HOME/datasets/glue_data
cd $HOME/datasets/
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks MRPC
```

#### SQuADv1.1 dataset preparation:
The data for SQuAD can be downloaded with the following links and should be saved into a directory. Specify path of the directory to demo script.
```
mkdir -p $HOME/datasets/Squad
cd $HOME/datasets/Squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
```
The pre-trained model will be downloaded the first time the demo is launched provided access to Internet is guaranteed.

## Training the Model

i. Fine-tune BERT base (Eager mode)

- Run BERT base fine-tuning on the GLUE MRPC dataset using FP32 data type:
```
  ./demo_bert --sub-command finetuning -m base -t MRPC -e 3  -b 64 -s 128 -p <dataset path>/MRPC --do-eval
```
- Run BERT base fine-tuning on the SQuAD dataset using FP32 data type:
```
  ./demo_bert --sub-command finetuning -m base -t SQUAD -b 12 -s 384 -e 2 -r 3e-05 -p <dataset path>/Squad --do-eval
```

ii. Fine-tune BERT large (Graph mode)

- Run BERT Large fine-tuning on the SQuAD dataset using BF16 mixed precision:
```
  ./demo_bert --sub-command finetuning -m large -t SQUAD -b 24 -s 384 -e 2 -r 3e-05 --mode graph --data-type bf16 -p <dataset path>/Squad --do-eval
 ```
- Run BERT Large fine-tuning on the SQuAD dataset using FP32 data type:
```
  ./demo_bert --sub-command finetuning -m large -t SQUAD -b 10 -s 384 -e 2 -r 3e-05 --data-type fp32 --mode graph -p <dataset path>/Squad --do-eval
 ```
- Run BERT Large fine-tuning on the MRPC dataset using FP32 data type:
```
  ./demo_bert --sub-command finetuning -m large -t MRPC -b 32 -s 128 -e 3 -r 3e-05 --data-type fp32 --mode graph -p <dataset path>/MRPC --do-eval
 ```

iii. Fine-tune BERT large (Eager mode)

- Run BERT Large fine-tuning on the MRPC dataset with FP32:
```
  ./demo_bert --sub-command finetuning -m large -t MRPC -b 32 -s 128 -e 3 -r 3e-05 --mode eager --data-type fp32 -p <dataset path>/MRPC --do-eval
 ```
- Run BERT Large fine-tuning on the SQuAD dataset with FP32:
```
  ./demo_bert --sub-command finetuning -m large -t SQUAD -b 10 -s 384 -e 2 -r 3e-05 --data-type fp32 --mode eager -p <dataset path>/Squad --do-eval
 ```


## Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-node demo.
```
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.14.0/ubuntu20.04/habanalabs/pytorch-installer:0.14.0-420
```
Before execution of the multi-node demo scripts, make sure all HLS-1 network interfaces are up. You can change the state of each network interface managed by the habanalabs driver using the following command:
```
sudo ip link set <interface_name> up
```
To identify if a specific network interface is managed by the habanalabs driver type, run:
```
sudo ethtool -i <interface_name>
```
Number of nodes can be configured using -w option in the demo script.
Use the following command to run the multinode demo on 8 cards (1 HLS) for bf16, BS24:
```
 ./demo_bert_dist --sub-command finetuning -t SQUAD -p <dataset path>/Squad -m large  -s 384 -b 24 -r 3e-05 --mode graph --data-type bf16 -e 2 -w 8 --do-eval --cache-dir <cache dir path>
```
Use the following command to run the multinode demo on 8 cards (1 HLS) for fp32, BS10:
```
 ./demo_bert_dist --sub-command finetuning -t SQUAD -p <dataset path>/Squad -m large  -s 384 -b 10 -r 3e-05 --mode graph --data-type fp32 -e 2 -w 8 --do-eval --cache-dir <cache dir path>
```


## Known Issues
1. MRPC finetuning: Final accuracy varies by 2% between different runs.

# Training Script Modifications
This section lists the training script modifications for the BERT models.

## BERT Large Pre-training
The following changes have been added to training & modeling scripts.

Modifications to the training script: (pretraining/run_pretraining.py)
1. Habana and CPU Device support
2. Saving checkpoint: Bring tensors to CPU and save
3. Torchscript jit trace mode support.
4. Pass position ids from training script to model.
5. int32 type instead of Long for input_ids, segment ids, position_ids and input_mask.
6. Habana BF16 Mixed precision support
7. Use Python version of LAMB optimizer.(from lamb.py)
8. Data loader: single worker thread, no pinned memory, skip last batch.
9. Conditional import of Apex modules
10. Support for distributed training on Habana device.
11. Use Fused LAMB optimizer
12. Loss computation brought outside of modeling script(pretraining/run_pretraining.py, pretraining/modeling.py)
13. Modified training script to use mpirun for distributed training. Introduced mpi barrier to sync the processes
14. Default allreduce bucket size set to 230MB for better performance in distributed training
15. Supports --tqdm_smoothing for controlling smoothing factor used for calculating iteration time.

Modifications to the modeling script: (pretraining/modeling.py)
1. On Non-Cuda devices, use the conventional linear and activation functions instead of combined linear activation
2. On Non-Cuda devices, use conventional nn.Layernorm instead of fused layernorm or layernorm using discrete ops.
3. Set embedding padding index to 0 explicitly.
4. Take position ids from training script rather than creating in the model.
5. Alternate select op implementation using index select and squeeze
6. Rewrote permute and view as flatten to enable better fusion in TorchScript trace mode
7. transpose_for_scores function modified to get a batchsize differently to enable the better fusion


## BERT Base and BERT Large Fine Tuning
The following changes have been added to scripts & source:

Modifications to the example training scripts (finetuning/examples dir):
1. Added Habana Device support(question-answering/run_squad.py)
2. Moved feature_index tensor to CPU(question-answering/run_squad.py)
3. Modifications for saving checkpoint: Bring tensors to CPU and save(question-answering/run_squad.py)
4. Modifications for adding support for Torchscript jit trace mode.(question-answering/run_squad.py)
5. Used int32 type instead of Long for input_ids, position_ids attention_mask, start_positions and
end_positions (question-answering/run_squad.py)
6. Distributed training : Use local barrier(question-answering/run_squad.py)
7. Introduced Habana BF16 Mixed precision to SQuAD script (question-answering/run_squad.py)
8. Use fused AdamW optimizer on Habana device (question-answering/run_squad.py)
9. Use fused clip norm for grad clipping on Habana device (question-answering/run_squad.py)
10. Modified training script to use mpirun for distributed training. Introduced mpi barrier to sync the processes(question-answering/run_squad.py)
11. Moved the params tensor list creation in clip norm wrapper to init func, so that list creation can be avoided in every iteration(question-answering/run_squad.py)
12. Gradients are used as views using gradient_as_bucket_view(question-answering/run_squad.py)
13. Changes for supporting HMP disable for optimizer.step(question-answering/run_squad.py)
14. Dropping the last batch if it is partial bacth size to effectively manage the memory(question-answering/run_squad.py)
15. Enabled the evaluation_during_training with fixes necessary(question-answering/run_squad.py)
16. Changes to optimize grad accumulation and zeroing of grads during the backward pass(question-answering/run_squad.py)
17. Default allreduce bucket size set to 230MB for better performance in distributed training(question-answering/run_squad.py)

Modifications for transformer source (finetuning/src/transformers dir):
1. Added Habana Device support (training_args.py)
2. Modifications for saving checkpoint: Bring tensors to CPU and save (modeling_utils.py; trainer.py)
3. Alternate select op implementation using index select and squeeze(modeling_bert.py)
4. Used div operator variant that takes both inputs as float (modeling_bert.py)
5. Checkpoint and tokenizer loading: Load optimizer from CPU; Load tokenizer from parent directory if available. (tokenization_utils_base.py,trainer.py)
6. Alternate addcdiv implementation in AdamW using discrete ops to avoid scalar scale factor(optimization.py)
7. Rewrote permute and view as flatten to enable better fusion in TorchScript trace mode (modeling_bert.py)
8. Used dummy tensor instead of ‘None’ for arguments like head_mask,inputs_embeds, start/end positions to be compatible with TorchScript trace mode(modeling_bert.py)
9. Distributed training : Use local barrier (trainer.py, training_args.py)
10. Distributed training : convert label_ids to int type (transformers/trainer.py)
11. Introduced Habana BF16 Mixed precision to SQuAD script (training_args.py)
12. Moved the params tensor list creation in clip norm wrapper to init func, so that list creation can be avoided in every iteration(transformers/trainer.py)
13. Used auto flush feature of summary writer instead explicit call(transformers/trainer.py)
14. Change for supporting HMP disable for optimizer.step(transformers/trainer.py)
15. Changes to optimize grad accumulation and zeroing of grads during the backward pass(transformers/trainer.py)
