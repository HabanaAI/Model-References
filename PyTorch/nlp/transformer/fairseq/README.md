# Table of Contents
- [Transformer for Pytorch](#transformer-for-pytorch)
  - [Model Overview](#model-overview)
- [Setup](#setup)
  - [Dataset preparation](#dataset-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluating BLEU Score](#evaluating-bleu-score)
  - [Multicard Training](#multicard-training)
- [Training Script Modifications](#training-script-modifications)
- [Known Issues](#known-issues)

# Transformer for Pytorch

The Transformer is a Neural Machine Translation (NMT) model which uses attention mechanism to boost training speed and overall accuracy. The model was initially introduced in Attention Is All You Need (https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

Training was done using WMT16 training data set and for validation/test WMT14 data set. Bleu score was calculated using sacrebleu utility (beam_size=4, length_penalty=0.6) on raw text translated data ( i.e the decoder tokenized output was detokenized using sacremoses before it was input to the sacrebleu calculator) .

The Transformer demos included in this release are Eager mode and Lazy mode training for max-tokens 4096 with FP32 and BF16 mixed precision, Multi card (1 server = 8 cards) Training support for Transformer with BF16 mixed precision in Lazy mode.

The Demo script is a wrapper for respective python training scripts. Additional environment variables are used in training scripts in order to achieve optimal results for each workload.

For more information about training deep learning models on Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Model Overview

The Transformer model uses standard NMT encoder-decoder architecture. Unlike other NMT models, it doesn't use recurrent connections and operates on fixed size context window. The encoder contains input embedding and positional encoding followed by stack is made up of N identical layers. Each layer is composed of the following sublayers:

  - Self-attention layer
  - Feedforward network (2 fully-connected layers)
The decoder stack is also made up output embedding and positional encoding of N identical layers. Each layer is composed of the sublayers:

  - Self-attention layer
  - Multi-headed attention layer combining encoder outputs with results from the previous self-attention layer.
  - Feedforward network (2 fully-connected layers)
The encoder uses self-attention to compute a representation of the input sequence. The decoder generates the output sequence one token at a time, taking the encoder output and previous decoder-outputted tokens as inputs. The model also applies embeddings on the input and output tokens, and adds a constant positional encoding. The positional encoding adds information about the position of each token.

The base training and validation scripts for Transformer are based on fairseq repo
https://github.com/pytorch/fairseq

The complete description of the Transformer architecture can be found in Attention Is All You Need paper.

# Setup
Please follow the instructions given in the following link for setting up the
environment including the `$PYTHON` environment variable: [Gaudi Setup and
Installation Guide](https://github.com/HabanaAI/Setup_and_Install). Please
answer the questions in the guide according to your preferences. This guide will
walk you through the process of setting up your system to run the model on
Gaudi.

In the docker container, clone this repository and switch to the branch that
matches your SynapseAI version. (Run the
[`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options)
utility to determine the SynapseAI version.)

```bash
git clone -b [SynapseAI version] https://github.com/HabanaAI/Model-References
```

## Dataset preparation

Follow steps 1 and 2 described in https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt

Install the required Python packages in the container:
```
 pip install -r Model-References/PyTorch/nlp/transformer/fairseq/requirements.txt
```
Extract the WMT'16 En-De data
copy the file wmt16_en_de.tar.gz to fairseq root directory
```
TEXT=Model-References/PyTorch/nlp/transformer/fairseq/wmt16_en_de_bpe32k
mkdir -p $TEXT
tar -xzvf wmt16_en_de.tar.gz -C $TEXT
```
Preprocess the dataset with a joined dictionary.
```
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
Alternatively dataset can be downloaded using seq2seq github project.

Data Generation:
```
git clone https://github.com/google/seq2seq.git
Update BASE_DIR=<path to seq2seq parent directory> in "bin/data/wmt16_en_de.sh"
bash bin/data/wmt16_en_de.sh
```
Above will generate the data in "/home/$user/nmt_data".

Data Preprocessing:
```
TEXT=/home/$user/nmt_data/wmt16_de_en
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
Copy the bpe.32000 file from $TEXT to the destdir with name bpe.code


## Training the Model
Clone the Model-References git.
Set up the data set as mentioned in the section "Dataset preparation".

```
cd Model-References/PyTorch/nlp/transfomer/fairseq
```
Run `$PYTHON demo_transformer.py -h` for command-line options.
For BLEU score calculation include --do-eval as an argument while using demo_transformer.py.

i. lazy mode, bf16 mixed precision, max-tokens 4096 for training and BLEU score calculation:
```
$PYTHON demo_transformer.py --max-update 30000 --mode lazy --data <data_path> --max-tokens 4096 --device hpu --optimizer adam --update-freq 13 --num-batch-buckets 10 --do-eval --data-type bf16
```

ii. lazy mode, fp32, max-tokens 4096 for training and BLEU score calculation:
```
$PYTHON demo_transformer.py --max-update 30000 --mode lazy --data <data_path> --max-tokens 4096 --device hpu --optimizer adam --update-freq 13 --num-batch-buckets 10 --do-eval --data-type fp32
```

iii. eager mode, bf16 mixed precision,  max-tokens 4096 for training and BLEU score calculation:
```
$PYTHON demo_transformer.py --max-update 30000 --mode eager --data <data_path> --max-tokens 4096 --device hpu --optimizer adam --update-freq 13 --num-batch-buckets 10 --do-eval --data-type bf16
```

## Evaluating BLEU Score
Evaluate BLEU score using the trained model:
```
sacrebleu -t wmt14 -l en-de --echo src \
| fairseq-interactive <data_path> --path <model_path> \
-s en -t de \
--batch-size 64 --buffer-size 1024 \
--beam 4 --lenpen 0.6 --remove-bpe --max-len-a 1.2 --max-len-b 10 \
--bpe subword_nmt --bpe-codes <data_path>/bpe.code --tokenizer moses --moses-no-dash-splits --use-habana --use-lazy-mode \
| tee out.txt \
| grep ^H- | cut -f 3- \
| sacrebleu -t wmt14 -l en-de
```

## Multicard Training
Follow the relevant steps under "Training the Model".
For BLEU score calculation include --do-eval as an argument while using demo_transformer.py.
To run multi-card demo, make sure the host machine has 512 GB of RAM installed. Modify the docker run command to pass 8 Gaudi cards to the docker container. This ensures the docker has access to all the 8 cards required for multi-card demo.

Use the following commands to run multicard training on 8 cards:

i. lazy mode, bf16 mixed precision, max-tokens 4096 for training and BLEU score calculation:
```
$PYTHON demo_transformer.py --max-update 30000 --mode lazy --data <data_path> --max-tokens 4096 --device hpu --optimizer adam --update-freq 13 --num-batch-buckets 10 --do-eval --data-type bf16 --world-size 8
```

```
export MASTER_ADDR="localhost"
export MASTER_PORT="12345"
mpirun -n 8 --bind-to core --map-by socket:PE=6 --rank-by core --report-bindings --allow-run-as-root $PYTHON fairseq_cli/train.py --arch transformer_wmt_en_de_big --lr 0.0005 --clip-norm 0.0 --dropout 0.3 --max-tokens 4096 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --update-freq 13 --save-interval-updates 1000 --keep-interval-updates 20 --log-format simple --log-interval 1 --share-all-embeddings --use-fused-adam --num-batch-buckets 10 --save-dir /tmp/fairseq_transformer_wmt_en_de_big/checkpoints --tensorboard-logdir /tmp/fairseq_transformer_wmt_en_de_big/tensorboard --maximize-best-checkpoint-metric --distributed-world-size 8 --bucket-cap-mb 230 --max-source-positions 256 --max-target-positions 256 --max-update 30000 --warmup-updates 4000 --warmup-init-lr 1e-07 --optimizer adam --adam-betas "(0.9, 0.98)" --lr-scheduler inverse_sqrt --no-epoch-checkpoints --use-lazy-mode --use-habana --bf16 /root/software/lfs/data/pytorch/Fairseq/transformer-LT/wmt16_en_de_bpe32k
```
ii. lazy mode, fp32,  max-tokens 4096 for training and BLEU score calculation:
```
$PYTHON demo_transformer.py --max-update 30000 --mode lazy --data <data_path> --max-tokens 4096 --device hpu --optimizer adam --update-freq 13 --num-batch-buckets 10 --do-eval --data-type fp32 --world-size 8
```

# Training Script Modifications
This section lists the training script modifications for the Transformer model.

The following changes have been added to training & modeling scripts.

1. Added support for Habana devices:

    a. Load Habana specific library.

    b. Required environment variables are defined for habana device.

    c. Added Habana BF16 Mixed precision support.

    d. Support for distributed training on Habana device.

    e. Added changes to support Lazy mode with required mark_step().

    f. Changes for dynamic loading of HCL library.

    g. Faireq-interactive evaluation is enabled with Habana device suport

2. Dataloader related changes:

    a. Data loader changes include single worker and no pinned memory

3. To improve performance:

    a. Added support for Fused Adam optimizer.

    b. Bucket size set to 230MB for better performance in distributed training.

    c. Number of buckets for training is set to 10 and values are fixed for Habana device.

4. Additional changes:

    a. Using number of buckets as 10 for training and for eval we use 3.

    b. Few pytorch ops are moved to cpu in the evaluation graph.

    c. Added Fused clip norm changes and not enabled in this release.

# Known Issues
1. Placing mark_step() arbitrarily may lead to undefined behaviour. Recommend to keep mark_step() as shown in provided scripts.
2. Evaluation takes more execution time due to cpu fallbacks and will be fixed in next release.
3. Total time to train is more which will be fixed in next release.

