# ALBERT

## Table of Contents

* [Model Overview](#model-overview)
* [Setup](#setup)
* [ALBERT Pre-Training](#albert-pre-training)
* [ALBERT Fine-Tuning](#albert-fine-tuning)
* [Downloading the datasets](#downloading-the-datasets)
* [Training the Model](#training-the-model)
* [Advanced](#advanced)
* [Examples](#examples)

## Model Overview

ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm by Google. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation.

Our implementation is a fork of [Google Research ALBERT](https://github.com/google-research/albert). Please visit [this page](../../../README.md#tensorflow-model-performance) for performance information.

## Setup

Please follow the instructions given in the following link for setting up the environment: [Gaudi Setup and Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please answer the questions in the guide according to your preferences. This guide will walk you through the process of setting up your system to run the model on Gaudi.

## ALBERT Pre-Training
- Suited for datasets:
   - `bookswiki`
   - `overfit`
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
    - `mrpc`: Microsoft Research Paraphrase Corpus (**MRPC**) is a paraphrase identification dataset, where systems aim to identify if two sentences are paraphrases of each other.
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

## Downloading the datasets
Follow the instructions in [https://github.com/HabanaAI/Model-References/tree/master/TensorFlow/nlp/bert#download-and-preprocess-the-datasets-for-pretraining-and-finetuning] in order to download Wikipedia and BookCorpus dataset.

Then use `data_preprocessing/create_training_data.py` to generate dataset files.

## Training the Model
Clone the repository and go to ALBERT directory:

```bash
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/TensorFlow/nlp/albert
pip install -r requirements.txt
```

If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/Model-References
```

## Examples
The training can be run with a custom python script `albert_demo.py` usage:

```python
python3 demo_albert.py --command <command> --model_variant <model> --data_type <data_type> --test_set <dataset_name> --dataset_path <path/to/dataset> --output_dir <model/data/path>
```

For example:

-  Single Gaudi card pretraining of albert Large, using bookswiki dataset on bfloat16 precision:
```python
python3 demo_albert.py --command pretraining --model_variant large --data_type bf16 --test_set bookswiki --dataset_path tensorflow_datasets/albert/bookswiki
```
-  Single Gaudi card finetuning of albert Large, using MRPC dataset on bfloat16 precision:
```python
python3 demo_albert.py --command finetuning --model_variant large --data_type bf16 --test_set mrpc --output_dir /root/tmp/albert_large --dataset_path tensorflow_datasets/albert/MRPC
```
-  Single Gaudi card finetuning of albert Large, using SQUAD dataset on bfloat16 precision:
```python
python3 demo_albert.py --command finetuning --model_variant large --data_type bf16 --test_set squad --output_dir /root/tmp/albert_large --dataset_path tensorflow_datasets/albert/squad
```
- 8 Gaudi cards finetuning of ALBERT Large in bfloat16 precision using SQuAD dataset on a single box (8 cards):
  ```bash
  cd /path/to/Model-References/TensorFlow/nlp/albert/

  python3 demo_albert.py \
     --command finetuning \
     --model_variant large \
     --data_type bf16 \
     --test_set squad \
     --dataset_path /software/data/tf/data/albert/tf_record/squad \
     --use_horovod 8 \
  2>&1 | tee ~/hlogs/albert_large_finetuning_bf16_squad_8_cards.txt
```
- 8 Gaudi cards finetuning of ALBERT Large in bfloat16 precision using SQuAD dataset on a K8s single box (8 cards):
```bash
  mpirun --allow-run-as-root \
         --bind-to core \
         --map-by socket:PE=6 \
         -np 8 \
         --tag-output \
         --merge-stderr-to-stdout \
         bash -c "cd /root/Model-References/TensorFlow/nlp/albert;\
                  python3 /root/Model-References/TensorFlow/nlp/albert/demo_albert.py \
                   --model_variant=large \
                   --command=finetuning \
                   --test_set=squad \
                   --data_type=bf16 \
                   --epochs=2 \
                   --batch_size=16 \
                   --max_seq_length=384 \
                   --learning_rate=3e-5 \
                   --output_dir=$HOME/tmp/squad_output_8cards/ \
                   --dataset_path=/software/data/tf/data/albert/tf_record/squad \
                   --use_horovod=1 \
                   --kubernetes_run=True" \
  2>&1 | tee ~/hlogs/albert_large_ft_squad_8cards.txt
```
The script automatically downloads the pre-trained model from https://storage.googleapis.com/albert_models/ the first time it is run in the docker container, as well as the dataset, if needed.

# Advanced
### Scripts
* `demo_albert.py`: Demo distributed luncher script, enables single and mutlinode training for both pretraining and finetuning tasks.
* `run_pretraining.py`: Script implementing pretraining task.
* `run_classifier.py`:  Script implementing MRPC task.
* `run_squad_v1.py`:  Script implementing SQUAD task.
