# BERT for PyTorch
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
These models are tested with Habana PyTorch docker container 0.13.0-380 for Ubuntu 20.04 and some dependencies contained within it.

### Requirements
* Docker version 19.03.12 or newer
* Sudo access to install required drivers/firmware
  OR
  If they are already installed, skip to #Pre-training or #Fine Tuning

### Install Drivers
Follow steps in the [Installation Guide](https://docs.habana.ai/projects/SynapeAI-Gaudi/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to install the drivers.

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
```
docker pull vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```
3. Run docker
```
docker run -td -v /dev:/dev --device=/dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```
4. Check name of your docker
 ```
docker ps
 ```
5. Run bash in your docker
 ```
docker exec -ti <NAME> bash 
 ```
6. In the docker container, clone the repository and go to PyTorch BERT directory:
 ```
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/nlp/bert
 ```
7. Run `./demo_bert -h` for command-line options

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
Run `./demo_bert -h` for command-line options
```
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

## Training Results
The following performance results were obtained by pre-training a BERT Large model in Graph mode:

| Model       | Dataset   | # Gaudi cards, Batch Size, Seq Length, Precision|  Throughput (Sequences/Sec)  |
|:------------|:-------------------|:------------------------:|:-------------------|
| BERT Large  | BooksCorpus, Wiki  | 1-card, BS=64, Seq=128 (phase1); BS8, Seq=512 (phase2), bf16   |   109.4 sps (phase1), 20 sps (phase2) |    
| BERT Large  | BooksCorpus, Wiki  | 1-card, BS=32, Seq=128 (phase1); BS4, Seq=512 (phase2), fp32   |    43 sps (phase1), 8.5 sps (phase2)|  

## Multinode Training 
Follow the relevant steps under "Training the Model". 
To run multi-node demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-node demo.
```
docker run -ti -v /dev:/dev --device=/dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
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
Use the following commands to run multinode training on 8 cards:

i. graph mode, bf16 mixed precision, per chip batch size of 64 for phase1 and 8 for phase2:
```
./demo_bert_dist --sub-command pretraining --data-type bf16 -b 64 8 -w 8 --data-dir <dataset path for phase1> <dataset path for phase2>
```
ii. graph mode, fp32 precision, per chip batch size of 32 for phase1 and 4 for phase2:
```
./demo_bert_dist --sub-command pretraining --data-type fp32 -b 32 4 -w 8 --data-dir <dataset path for phase1> <dataset path for phase2>
```

## Multinode Training Results
The following performance results were obtained by pre-training a BERT Large model in Graph mode:

| Model       | Dataset   | # Gaudi cards, Batch Size, Seq Length, Precision|  Throughput (Sequences/Sec)  |   Scaling |
|:------------|:-------------------|:------------------------:|:-------------------|:--------------------|
| BERT Large  | BooksCorpus, Wiki  | 8-cards, BS=64, Seq=128 (phase1), BS8, Seq=512 (phase2), bf16   |   105.6 sps (phase1), 17.12 sps (phase2) | 96.4% for phase1 and 85.6% for phase2 | 
| BERT Large  | BooksCorpus, Wiki  | 8-cards, BS=32, Seq=128 (phase1), BS4, Seq=512 (phase2), fp32   |   40.64 sps (phase1) and 7.4 sps (phase2) | 94.5% for phase1 and 87% for phase2 |

## Known Issues
- Pretraining with FP32, for BS8 for phase2 results in OOM error.

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
```
docker pull vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```
3. Run docker
```
docker run -td -v /dev:/dev --device=/dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
```
4. Check name of your docker
 ```
docker ps
 ```
5. Run bash in your docker
 ```
docker exec -ti <NAME> bash 
 ```
6. In the docker container, clone the repository and go to PyTorch BERT directory:
 ```
git clone https://github.com/HabanaAI/Model-References
cd Model-References/PyTorch/nlp/bert
 ```
7. Run `./demo_bert -h` for command-line options

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

## Training Results
The following performance and accuracy results were obtained by fine-tuning pre-trained BERT Large model with SQuAD dataset in Graph mode.

| Model       | Dataset   | # Gaudi cards, Batch Size, Seq Length, Precision|  Throughput (Sequences/Sec)  |  Accuracy  |
|:------------|:----------|:------------------------:|:-------------------:|------------:|
| BERT Large  | SQuADv1.1    | 1-card, BS=24, Seq=384, bf16     |   30.2 sps | 87.5% |
| BERT Large  | SQuADv1.1    | 1-card, BS=10, Seq=384, fp32     |   11.8 sps | 86.6% |

## Multinode Training
To run multi-node demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-node demo.
```
docker run -ti -v /dev:/dev --device=/dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug -v $HOME/datasets/:/root/dataset --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu20.04/habanalabs/pytorch-installer:0.13.0-380
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
## Multinode Training Results 
The following results were obtainged for BERT fine tuning in Graph mode: 

| Model       | Dataset   |# Gaudi cards, Batch Size, Seq Length, Precision|  Throughput (Sequences/Sec)  |Time to Train (Hrs)  |  Scaling %  |  Accuracy |
|:------------|:----------|:------------------------:|:-------------------|:-------------------:|------------:|:---------------|
| BERT Large   | SQuADv1.1    | 8-cards, BS=24, Seq=384, bf16 | 26.11 sps | 22 mins | 86.4% |  87.5% |
| BERT Large   | SQuADv1.1    | 8-cards, BS=10, Seq=384, fp32 | 10.2 sps | 44 mins | 86.6% | 86.6% |

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

Modifications to the modeling script: (pretraining/modeling.py)
1. On Non-Cuda devices, use the conventional linear and activation functions instead of combined linear activation 
2. On Non-Cuda devices, use conventional nn.Layernorm instead of fused layernorm or layernorm using discrete ops.
3. Set embedding padding index to 0 explicitly.
4. Take position ids from training script rather than creating in the model.
5. Alternate select op implementation using index select and squeeze
6. Rewrote permute and view as flatten to enable better fusion in TorchScript trace mode 

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

Modifications for transformer source (finetuning/src/transformers dir):
1. Added Habana Device support (training_args.py)
2. Modifications for saving checkpoint: Bring tensors to CPU and save (modeling_utils.py; trainer.py)
3. Alternate select op implementation using index select and squeeze(modeling_bert.py)
4. Used div operator variant that takes both inputs as float (modeling_bert.py)
5. Checkpoint and tokenizer loading: Load optimizer from CPU; Load tokenizer from parent directory if available. (tokenization_utils_base.py,trainer.py)
6. Alternate addcdiv implementation in AdamW using discrete ops to avoid scalar scale factor(optimization.py)
7. Rewrote permute and view as flatten to enable better fusion in TorchScript trace mode (modeling_bert.py)
8. Used dummy tensor instead of ‘None’ for arguments like head_mask,inputs_embeds, start/end positions to be compatible with TorchScript trace mode(modeling_bert.py
9. Distributed training : Use local barrier (trainer.py, training_args.py)
10. Distributed training : convert label_ids to int type (transformers/trainer.py)
11. Introduced Habana BF16 Mixed precision to SQuAD script (training_args.py)
