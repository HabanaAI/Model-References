# Transformer for Pytorch

This directory provides a script and recipe to train a Transformer model for PyTorch to achieve state of the art accuracy, and is tested and maintained by Habana. For further information on performance, refer to [Habana Model Performance Data page](https://developer.habana.ai/resources/habana-training-models/#performance).

For further information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents
- [Model References](../../../../README.md)
- [Model Overview](#model-overview)
- [Setup](#setup)
- [Traning Examples](#training-examples)
- [Supported Configurations](#supported-configurations)
- [Changelog](#changelog)
- [Known Issues](#known-issues)


## Model Overview

This is a PyTorch implementation of the Transformer model training and validation scripts based on [Fairseq repository](https://github.com/pytorch/fairseq).

The Transformer is a Neural Machine Translation (NMT) model which uses attention mechanism to boost training speed and overall accuracy. The model was initially introduced in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017). The training was implemented using WMT16 training dataset, whereas for validation/test WMT14 dataset. BLEU score was calculated using sacrebleu utility (beam_size=4, length_penalty=0.6) on raw text translated data (i.e the decoder tokenized output was detokenized using sacremoses before it was input to the sacrebleu calculator).

The Transformer demos included in this release are Eager mode and Lazy mode training for max-tokens 4096 with FP32 and BF16 mixed precision. Multi-card (1 server = 8 cards) training is supported for Transformer with BF16 mixed precision in Lazy mode.

### Model Architecture

The Transformer model uses standard NMT encoder-decoder architecture. Unlike other NMT models, not only it does not use recurrent connections, but also operates on fixed size context window.
The encoder contains input embedding and positional encoding followed by stack which is made up of N identical layers. Each layer is composed of the following sub-layers:

  - Self-attention layer
  - Feedforward network (2 fully-connected layers)

The decoder stack is also made up output embedding and positional encoding of N identical layers. Each layer is composed of the following sub-layers:

  - Self-attention layer
  - Multi-headed attention layer combining encoder outputs with results from the previous self-attention layer.
  - Feedforward network (2 fully-connected layers)

The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs. The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

## Setup

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/GAUDI_Installation_Guide.html) to set up the
environment including the `$PYTHON` environment variable. The guide will walk you through the process of setting up your system to run the model on Gaudi.

### Clone Habana Model-References

In the docker container, clone this repository and switch to the branch that matches your SynapseAI version. You can run the [`hl-smi`](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the SynapseAI version.

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References /root/Model-References
```

**Note:** If Model-References repository path is not in the PYTHONPATH, make sure you update it:
```bash
export PYTHONPATH=$PYTHONPATH:/root/Model-References
```

### Download the Dataset

1. To download the dataset, use seq2seq GitHub project:
```
git clone https://github.com/google/seq2seq.git
cd seq2seq
```
2. Update BASE_DIR=<path to seq2seq parent directory> in `bin/data/wmt16_en_de.sh` and run the following:

**NOTE:** The data will be generated in the `OUTPUT_DIR`.
```
export OUTPUT_DIR=/data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k
bash bin/data/wmt16_en_de.sh
```

### Install Model Requirements

1. Install the required Python packages in the container:
```
 pip install -r Model-References/PyTorch/nlp/nmt/fairseq/requirements.txt
```
2. Install Fairseq module:
```
pip install Model-References/PyTorch/nlp/nmt/fairseq/.
```

### Pre-process the Data

1. To pre-process the data, run the following:

```
TEXT=/data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
```

2. Copy the `bpe.32000` file from `$TEXT` to the destdir with name `bpe.code`:
```
cp $TEXT/bpe.32000 $TEXT/bpe.code
```

## Training Examples

**NOTE:** The training examples are applicable for first-gen Gaudi and **Gaudi2**

### Single Card and Multi-Card Training Examples

- To see all CommandLine options, run `$PYTHON fairseq_cli/train.py  -h`.
- For BLEU score calculation, use sacreBLEU module after the training.

**Run training on 1 HPU:**

- 1 HPU in lazy mode, BF16 mixed precision, max-tokens 4096 for training and BLEU score calculation:
  ```
  # training
  $PYTHON fairseq_cli/train.py /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --arch=transformer_wmt_en_de_big --lr=0.0005 --clip-norm=0.0 --dropout=0.3 --max-tokens=4096 --weight-decay=0.0 --criterion=label_smoothed_cross_entropy --label-smoothing=0.1 --update-freq=13 --save-interval-updates=3000 --save-interval=10 --validate-interval=20 --keep-interval-updates=20 --log-format=simple --log-interval=20 --share-all-embeddings --num-batch-buckets=10 --save-dir=/tmp/fairseq_transformer_wmt_en_de_big/checkpoint --tensorboard-logdir=/tmp/fairseq_transformer_wmt_en_de_big/tensorboard --maximize-best-checkpoint-metric --max-source-positions=256 --max-target-positions=256 --max-update=30000 --warmup-updates=4000 --warmup-init-lr=1e-07 --lr-scheduler=inverse_sqrt --no-epoch-checkpoints --bf16 --optimizer=adam --use-fused-adam --adam-betas="(0.9, 0.98)" --use-habana
  # evaluation
  sacrebleu -t wmt14 -l en-de --echo src | fairseq-interactive /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --path /tmp/fairseq_transformer_wmt_en_de_big/checkpoint/checkpoint_last.pt -s en -t de --batch-size 32 --buffer-size 1024 --beam 4 --lenpen 0.6 --remove-bpe --max-len-a 1.2 --max-len-b 10 --bpe subword_nmt --bpe-codes /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k/bpe.code --tokenizer moses --moses-no-dash-splits --use-habana | tee /tmp/.eval_out.txt | grep ^H- | cut -f 3- | sacrebleu -t wmt14 -l en-de
  ```

- 1 HPU in lazy mode, FP32, max-tokens 4096 for training and BLEU score calculation:

  ```
  # training
  $PYTHON fairseq_cli/train.py /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --arch=transformer_wmt_en_de_big --lr=0.0005 --clip-norm=0.0 --dropout=0.3 --max-tokens=4096 --weight-decay=0.0 --criterion=label_smoothed_cross_entropy --label-smoothing=0.1 --update-freq=13 --save-interval-updates=3000 --save-interval=10 --validate-interval=20 --keep-interval-updates=20 --log-format=simple --log-interval=20 --share-all-embeddings --num-batch-buckets=10 --save-dir=/tmp/fairseq_transformer_wmt_en_de_big/checkpoint --tensorboard-logdir=/tmp/fairseq_transformer_wmt_en_de_big/tensorboard --maximize-best-checkpoint-metric --max-source-positions=256 --max-target-positions=256 --max-update=30000 --warmup-updates=4000 --warmup-init-lr=1e-07 --lr-scheduler=inverse_sqrt --no-epoch-checkpoints --optimizer=adam --use-fused-adam --adam-betas="(0.9, 0.98)" --use-habana
  # evaluation
  sacrebleu -t wmt14 -l en-de --echo src | fairseq-interactive /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --path /tmp/fairseq_transformer_wmt_en_de_big/checkpoint/checkpoint_last.pt -s en -t de --batch-size 32 --buffer-size 1024 --beam 4 --lenpen 0.6 --remove-bpe --max-len-a  1.2 --max-len-b 10 --bpe subword_nmt --bpe-codes /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k/bpe.code --tokenizer moses --moses-no-dash-splits --use-habana | tee /tmp/.eval_out.txt | grep ^H- | cut -f 3- | sacrebleu -t wmt14 -l en-de
  ```

- 1 HPU in eager mode, BF16 mixed precision, max-tokens 4096 for training and BLEU score calculation:

  ```
  # training
  $PYTHON fairseq_cli/train.py /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --arch=transformer_wmt_en_de_big --lr=0.0005 --clip-norm=0.0 --dropout=0.3 --max-tokens=4096 --weight-decay=0.0 --criterion=label_smoothed_cross_entropy --label-smoothing=0.1 --update-freq=13 --save-interval-updates=3000 --save-interval=10 --validate-interval=20 --keep-interval-updates=20 --log-format=simple --log-interval=20 --share-all-embeddings --num-batch-buckets=10 --save-dir=/tmp/fairseq_transformer_wmt_en_de_big/checkpoint --tensorboard-logdir=/tmp/fairseq_transformer_wmt_en_de_big/tensorboard --maximize-best-checkpoint-metric --max-source-positions=256 --max-target-positions=256 --max-update=30000 --warmup-updates=4000 --warmup-init-lr=1e-07 --lr-scheduler=inverse_sqrt --no-epoch-checkpoints --bf16 --optimizer=adam --use-fused-adam --adam-betas="(0.9, 0.98)" --use-habana --use-lazy-mode='False'
  # evaluation
  sacrebleu -t wmt14 -l en-de --echo src | fairseq-interactive /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --path /tmp/fairseq_transformer_wmt_en_de_big/checkpoint/checkpoint_last.pt -s en -t de --batch-size 32 --buffer-size 1024 --beam 4 --lenpen 0.6 --remove-bpe --max-len-a  1.2 --max-len-b 10 --bpe subword_nmt --bpe-codes /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k/bpe.code --tokenizer moses --moses-no-dash-splits --use-habana --use-lazy-mode='False' | tee /tmp/.eval_out.txt | grep ^H- | cut -f 3- | sacrebleu -t wmt14 -l en-de
  ```

**Run training on 8 HPUs:**

To run multi-card demo, make sure to set the following prior to the training:
- The host machine has 512 GB of RAM installed.

**NOTE:** mpirun map-by PE attribute value may vary on your setup. For the recommended calculation, refer to the instructions detailed in [mpirun Configuration](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Scaling_Guide/DDP_Based_Scaling.html#mpirun-configuration).

- 8 HPUs in lazy mode, BF16 mixed precision, max-tokens 4096 for training and BLEU score calculation:

  ```
  # training
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345"
  mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON fairseq_cli/train.py /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --arch=transformer_wmt_en_de_big --lr=0.0005 --clip-norm=0.0 --dropout=0.3 --max-tokens=4096 --weight-decay=0.0 --criterion=label_smoothed_cross_entropy --label-smoothing=0.1 --update-freq=13 --save-interval-updates=3000 --save-interval=10 --validate-interval=20 --keep-interval-updates=20 --log-format=simple --log-interval=1 --share-all-embeddings --num-batch-buckets=10 --save-dir=/tmp/fairseq_transformer_wmt_en_de_big/checkpoint --tensorboard-logdir=/tmp/fairseq_transformer_wmt_en_de_big/tensorboard --maximize-best-checkpoint-metric --max-source-positions=256 --max-target-positions=256 --max-update=30000 --warmup-updates=4000 --warmup-init-lr=1e-07 --lr-scheduler=inverse_sqrt --no-epoch-checkpoints --eval-bleu-args='{"beam":4,"max_len_a":1.2,"max_len_b":10}' --eval-bleu-detok=moses --eval-bleu-remove-bpe --eval-bleu-print-samples --eval-bleu --bf16 --optimizer=adam --use-fused-adam --adam-betas="(0.9, 0.98)" --use-habana --distributed-world-size=8 --bucket-cap-mb=230
  # evaluation
  sacrebleu -t wmt14 -l en-de --echo src | fairseq-interactive /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --path /tmp/fairseq_transformer_wmt_en_de_big/checkpoint/checkpoint_last.pt -s en -t de --batch-size 32 --buffer-size 1024 --beam 4 --lenpen 0.6 --remove-bpe --max-len-a  1.2 --max-len-b 10 --bpe subword_nmt --bpe-codes /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k/bpe.code --tokenizer moses --moses-no-dash-splits --use-habana | tee /tmp/.eval_out.txt | grep ^H- | cut -f 3- | sacrebleu -t wmt14 -l en-de
  ```
- 8 HPUs in lazy mode, FP32,  max-tokens 4096 for training and BLEU score calculation:

  ```
  # training
  export MASTER_ADDR="localhost"
  export MASTER_PORT="12345"
  mpirun -n 8 --bind-to core --map-by slot:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON fairseq_cli/train.py /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --arch=transformer_wmt_en_de_big --lr=0.0005 --clip-norm=0.0 --dropout=0.3 --max-tokens=4096 --weight-decay=0.0 --criterion=label_smoothed_cross_entropy --label-smoothing=0.1 --update-freq=13 --save-interval-updates=3000 --save-interval=10 --validate-interval=20 --keep-interval-updates=20 --log-format=simple --log-interval=1 --share-all-embeddings --num-batch-buckets=10 --save-dir=/tmp/fairseq_transformer_wmt_en_de_big/checkpoint --tensorboard-logdir=/tmp/fairseq_transformer_wmt_en_de_big/tensorboard --maximize-best-checkpoint-metric --max-source-positions=256 --max-target-positions=256 --max-update=30000 --warmup-updates=4000 --warmup-init-lr=1e-07 --lr-scheduler=inverse_sqrt --no-epoch-checkpoints --optimizer=adam --use-fused-adam --adam-betas="(0.9, 0.98)" --use-habana --distributed-world-size=8 --bucket-cap-mb=230
  # evaluation
  sacrebleu -t wmt14 -l en-de --echo src | fairseq-interactive /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k --path /tmp/fairseq_transformer_wmt_en_de_big/checkpoint/checkpoint_last.pt -s en -t de --batch-size 32 --buffer-size 1024 --beam 4 --lenpen 0.6 --remove-bpe --max-len-a  1.2 --max-len-b 10 --bpe subword_nmt --bpe-codes /data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k/bpe.code --tokenizer moses --moses-no-dash-splits --use-habana | tee /tmp/.eval_out.txt | grep ^H- | cut -f 3- | sacrebleu -t wmt14 -l en-de
  ```

## Supported Configurations

| Validated on | SynapseAI Version | PyTorch Version | Mode |
|--------|-------------------|-----------------|----------------|
| Gaudi   | 1.8.0             | 1.13.1          | Training |
| Gaudi2  | 1.8.0             | 1.13.1          | Training | 

## Changelog

### 1.6.0
1. Forcing softmax operator always with dtype fp32 is removed.
1. Added support for 1 and 8 card training on **Gaudi2**

### 1.5.0
1. Simplified the distributed initialization.
2. Removed unsupported examples.
3. In Evaluation graph, removed cpu fallbacks and reduced the frequency of mark_step() call.
4. Removed Fused clip norm.

### 1.4.0
1. Lazy mode is set as default execution mode,for eager mode set --use-lazy-mode as False.
2. Removed redundant import of htcore from model scripts.
3. Added distributed barrier to synchronize the host process.
4. setup_requirement for numpy version is set to 1.22.2 for python version 3.8.

### 1.3.0
1. Most of the pytorch ops are moved to hpu in the evaluation graph.
2. Changes related to single worker thread is removed.

### 1.2.0
1. Enabled HCCL flow for distributed training.
2. Data dependent control loops are avoided in encoder and decoder.
3. Reduce and stats collection based on every log_interval steps and log_interval is set to 20 in single node.
4. Evaluation with Bleu score is added part of training and reduced the frequency of evaluation
5. Multicard training is issue is fixed with data prepared from seq2seq github.

### Training Script Modifications
This section lists the training script modifications for the Transformer model.

- Added support for Habana devices:
  - Load Habana specific library.
  - Required environment variables are defined for habana device.
  - Support for distributed training on Habana device.
  - Added changes to support Lazy mode with required mark_step().
  - Changes for dynamic loading of HCCL library.
  - Fairseq-interactive evaluation is enabled with Habana device support.
  - Added distributed barrier to synchronize the host process

- To improve performance:
  - Added support for Fused Adam optimizer.
  - Bucket size set to 230MB for better performance in distributed training.
  - Number of buckets for training is set to 10 and values are fixed for Habana device.
  - Data dependent control loops are avoided in encoder and decoder.
  - Reduce and stats collection based on every log_interval steps.(ex:log_interval=20 for single card).
  - To get BLEU score accuracy of 27.5 set "--max-update" as "15000" which will reduce the total "time to train" by ~50%.

## Known Issues
1. Placing `mark_step()` arbitrarily may lead to undefined behavior. It is recommended to keep `mark_step()` as shown in the provided scripts.
2. Evaluation of execution is not fully optimal, reduced frequency of execution is recommended in this release.
3. Only scripts and configurations mentioned in this README are supported and verified.
