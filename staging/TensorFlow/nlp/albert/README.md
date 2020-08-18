# ALBERT Demo

ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm by Google. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation.

**THIS IS A PREVIEW OF OUR ALBERT IMPLEMENTATION. It is functional but still in progress.** Our implementation is a fork of [Google Research ALBERT](https://github.com/google-research/albert)

## Quick Setup
- Get the Habana TensorFlow docker image for ubuntu18.04:
```bash
docker pull vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```
- Run the docker container:
```bash
docker run -it --device=/dev:/dev -v /dev:/dev -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice -v /sys/kernel/debug:/sys/kernel/debug --net=host vault.habana.ai/gaudi-docker/0.13.0/ubuntu18.04/habanalabs/tensorflow-installer:0.13.0-380
```
- In the docker container:
```bash
git clone https://github.com/HabanaAI/Model-References.git
cd Model-References/staging/TensorFlow/nlp/albert
pip install -r requirements.txt
```

## ALBERT Pre-Training
- Suited for datasets:
    - `overfit`
    - provided by user (use -i to specify path to dataset)
- Default hyperparameters:
    - dataset: overfit
    - eval_batch_size: 8
    - max_seq_length: 128
    - optimizer: lamb
    - learning_rate: 0.00176
    - num_train_steps: 200
    - num_warmup_steps: 10
    - save_checkpoints_steps: 5000
    - lower_case: true
    - do_train: true
    - do_eval: true
    - use_einsum: false
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

## Running the Training and Evaluation
```bash
./demo_albert -h  # for help
```
- The script will automatically download the pre-trained model from https://storage.googleapis.com/albert_models/ the first time it is run in the docker container.
- Default batch sizes:
    - base: 64 (bf16), 32 (fp32)
    - large: 32 (bf16), 16 (fp32)

## Training Results

### Performance
| Model        | Task & Dataset            | # Gaudi cards, Batch Size, Seq Length, Precision | Throughput (Sentences/Sec) | Command Line                                                         |
|:-------------|---------------------------|:------------------------------------------------:|:--------------------------:|:---------------------------------------------------------------------|
| ALBERT Base  | Fine-Tuning with SQuAD    | 1-card, BS=32, Seq=384, bf16                     | 150.1                      | `./demo_albert finetuning -d bf16 -m base -t squad -b 32 -s 384`     |
| ALBERT Base  | Fine-Tuning with SQuAD    | 1-card, BS=32, Seq=384, fp32                     | 60.5                       | `./demo_albert finetuning -d fp32 -m base -t squad -b 32 -s 384`     |
| ALBERT Large | Pre-Training with overfit | 1-card, BS=32, Seq=128, bf16                     | 162.3                      | `./demo_albert pretraining -d bf16 -m large -t overfit -b 32 -s 128` |
| ALBERT Large | Pre-Training with overfit | 1-card, BS=32, Seq=128, fp32                     | 58.8                       | `./demo_albert pretraining -d fp32 -m large -t overfit -b 32 -s 128` |


### Accuracy
| Model        | Task & Dataset          | # Gaudi cards, Batch Size, Seq Length, Precision | Accuracy (F1/EM)           | Command Line                                                            |
|:-------------|-------------------------|:------------------------------------------------:|:--------------------------:|:------------------------------------------------------------------------|
| ALBERT Base  | Fine-Tuning with SQuAD  | 1-card, BS=32, Seq=384, bf16                     | 88.1/80.6                  | `./demo_albert finetuning -d bf16 -m base -t squad -b 32 -s 384 -e 3.0` |
| ALBERT Base  | Fine-Tuning with SQuAD  | 1-card, BS=32, Seq=384, fp32                     | 87.2/79.4                  | `./demo_albert finetuning -d fp32 -m base -t squad -b 32 -s 384 -e 3.0` |
